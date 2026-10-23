"""Embedding API for google/embeddinggemma-2.

POST /embed  multipart/form-data with:
  request  (optional) JSON: {"text": "...", "prompt_name": null, "title": null, "dimensions": 768,
                             "normalize": false}
  file     (repeatable) raw file bytes; image/audio/video comes from the filename's extension
e.g. curl host:11433/embed -F 'request={"text": "see <|image|>"}' -F file=@photo.jpg
Returns {"modality": ..., "dimensions": N, "embedding": [floats]}.

POST /embed/batch  many items per request, for throughput. See README.md.

Run: .venv/bin/python app.py  (settings come from the environment or .env; see .env.example)
"""

import math
import os
import shutil
import tempfile
import threading
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env before the other imports: some libraries (e.g. huggingface_hub's HF_HOME) read
# their settings at import time. Variables already set in the environment win over .env.
load_dotenv(Path(__file__).with_name(".env"))

import av
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from starlette.datastructures import UploadFile as StarletteUploadFile
from sentence_transformers.base.modules.normalize import Normalize
from torchcodec.decoders import AudioDecoder



def env(name: str, default: str) -> str:
    """A setting from the environment or .env; unset and empty both mean the default."""
    return os.environ.get(name, "").strip() or default


# Deployment settings; see .env.example for what each one does.
HOST = env("HOST", "0.0.0.0")
PORT = int(env("PORT", "11433"))
MODEL_ID = env("MODEL_ID", "google/embeddinggemma-2")
DEVICE = env("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# float16 can silently produce NaN embeddings with this model, so it's refused outright.
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
if env("DTYPE", "float32") not in _DTYPES:
    raise SystemExit(f"DTYPE must be one of {sorted(_DTYPES)}, got {env('DTYPE', '')!r}")
DTYPE = _DTYPES[env("DTYPE", "float32")]
MAX_TOKENS = int(env("MAX_TOKENS", "8192"))
BATCH_TOKENS = int(env("BATCH_TOKENS", "16384"))
MAX_BATCH_ITEMS = int(env("MAX_BATCH_ITEMS", "256"))
IDLE_CACHE_MB = int(env("IDLE_CACHE_MB", "1024"))

# Fixed properties of the model and its processors, not settings.
Dimensions = Literal[768, 512, 256, 128]  # Matryoshka sizes the model was trained for
AUDIO_SAMPLE_RATE = 16_000
MEDIA_MODALITIES = ("image", "audio", "video")
# Explicit table rather than mimetypes: it misses .webp, maps .3gp to audio and .ts to
# text, and varies with the host's /etc/mime.types.
EXTENSION_MODALITIES = {
    **dict.fromkeys(
        ["jpg", "jpeg", "jfif", "jpe", "pjpeg", "png", "gif", "webp", "bmp", "tif", "tiff", "avif", "heic", "heif"],
        "image",
    ),
    **dict.fromkeys(["wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus"], "audio"),
    **dict.fromkeys(["mp4", "m4v", "mov", "avi", "mkv", "webm", "mpeg", "mpg", "3gp", "ogv", "wmv", "flv", "ts"], "video"),
}

# Token costs, used to keep each input within MAX_TOKENS. Nothing in the model enforces the
# budget, so without it large requests run but can exhaust GPU memory (100 images ~= 23 GB).
# These are upper bounds measured against the processor.
IMAGE_TOKENS = 280
VIDEO_TOKENS_PER_FRAME = 140
AUDIO_TOKENS_PER_SECOND = 25
# The audio feature extractor silently truncates at 30s (480k samples) by default.
# Raise that so the token budget is the only limit (~327s at 8192 tokens, per the model card).
AUDIO_MAX_SAMPLES = math.ceil(MAX_TOKENS / AUDIO_TOKENS_PER_SECOND) * AUDIO_SAMPLE_RATE

# One request on the GPU at a time: concurrent large batches would otherwise run out of
# memory, and batching already keeps the GPU busy.
GPU_LOCK = threading.Lock()

# Lets Pillow open HEIC/HEIF (e.g. iPhone photos); Pillow handles the other image formats itself.
register_heif_opener()

model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer(MODEL_ID, device=DEVICE, model_kwargs={"torch_dtype": DTYPE})
    # The checkpoint's pipeline ends with a Normalize module; drop it so vectors are
    # returned raw (mean-pooled), not unit length.
    for i in reversed(range(len(model))):
        if isinstance(model[i], Normalize):
            del model[i]
    yield


app = FastAPI(
    title="EmbeddingGemma API",
    version="1.0",
    description="Embeds text, images, audio and video, alone or combined, with "
    "[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2). "
    "Requests are `multipart/form-data`: a JSON `request` part with the options, plus one `file` part per "
    "file. A file's type comes from its extension.\n\n"
    f"**Supported extensions:** {', '.join(sorted(EXTENSION_MODALITIES))}\n\n"
    "**Token budget:** each input shares an 8,192-token budget. An image costs 280 tokens, audio 25 per second "
    "(up to ~327 s), video 140 per frame. Video is sampled at 1 frame per second, using whatever budget the rest "
    "of the input leaves; longer videos get frames spread evenly across their whole length.",
    lifespan=lifespan,
)


class TextOptions(BaseModel):
    """Text and its formatting, shared by /embed and each /embed/batch item."""

    # Reject misspelled options instead of silently ignoring them.
    model_config = {"extra": "forbid"}

    text: str | None = Field(
        None,
        description="Text to embed. May contain `<|image|>`, `<|audio|>` and `<|video|>` placeholders to position "
        "files within the text; if any are used, there must be exactly one per file of that type, filled in upload "
        "order. Without placeholders, files are added after the text: images, then audio, then video.",
        examples=["Waterproof trail shoes. <|image|>"],
    )
    prompt_name: str | None = Field(
        None,
        description="Task prefix added to the text (never to files), e.g. `SearchQuery` → "
        "`task: search result | query: {text}`. For search, use a query prompt (`SearchQuery`, "
        "`QuestionAnswering`, `FactChecking`, `CodeRetrieval`) on queries and `Document` or `title` on the "
        "documents being searched. For comparing like with like, use the same one of `Classification`, "
        "`Clustering` or `SentenceSimilarity` on everything. Ignored when there's no text.",
        examples=["SearchQuery"],
    )
    title: str | None = Field(
        None,
        description="Formats the text as a titled document: `title: {title} | text: {text}`. Can't be combined "
        "with a query `prompt_name`. Use `prompt_name: \"Document\"` instead for documents without a title.",
        examples=["Trail shoe"],
    )


class VectorOptions(BaseModel):
    model_config = {"extra": "forbid"}

    dimensions: Dimensions = Field(
        768,
        description="Length of the returned vector. Shorter vectors are the first N values of the full one; "
        "quality is close to full down to 256 and drops noticeably at 128.",
    )
    normalize: bool = Field(
        False,
        description="`true` scales the vector to unit length, after shortening it to `dimensions`. With `false`, "
        "the raw vector is returned; compare raw vectors with cosine similarity, not a plain dot product.",
    )


class EmbedOptions(VectorOptions, TextOptions):
    """The JSON `request` part of /embed. Files are sent as separate `file` parts."""

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Waterproof trail shoes. <|image|>", "prompt_name": "SearchQuery", "dimensions": 768}]
        }
    }


class ItemOptions(TextOptions):
    """One input in a batch."""

    files: list[str] = Field(
        [],
        description="Filenames of uploaded `file` parts that belong to this item. A file can be used by "
        "several items.",
        examples=[["shoe.jpg"]],
    )


class BatchOptions(VectorOptions):
    """The JSON `request` part of /embed/batch. `dimensions` and `normalize` apply to every item."""

    items: list[ItemOptions] = Field(
        description="Inputs to embed, 1 to MAX_BATCH_ITEMS (default 256). Each follows the same rules and "
        "token budget as a single /embed request."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {"text": "waterproof trail shoes", "prompt_name": "SearchQuery"},
                        {"title": "Trail shoe", "text": "Grippy sole <|image|>", "files": ["shoe.jpg"]},
                        {"files": ["demo.mp4"]},
                    ],
                    "dimensions": 768,
                }
            ]
        }
    }


class EmbedResponse(BaseModel):
    modality: str = Field(
        description="Input types that went into the vector, `+`-joined and sorted, e.g. `text`, `image+text`, "
        "`audio+image+text+video`.",
        examples=["image+text"],
    )
    dimensions: int = Field(description="Length of `embedding`.", examples=[768])
    embedding: list[float] = Field(description="The vector. Raw unless `normalize` was `true`.")


class BatchResult(BaseModel):
    modality: str = Field(description="Input types that went into this item's vector.", examples=["text"])
    embedding: list[float] = Field(description="This item's vector. Raw unless `normalize` was `true`.")


class BatchResponse(BaseModel):
    dimensions: int = Field(description="Length of every `embedding`.", examples=[768])
    results: list[BatchResult] = Field(description="One result per item, in the order the items were sent.")


class ErrorResponse(BaseModel):
    detail: str | dict = Field(
        description="What went wrong. A string for most errors; for 413, an object with `message`, "
        "`estimated_tokens` (per input type) and, in batches, the failing `item` index."
    )


ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Nothing to embed, invalid `request` JSON or option value, unknown `prompt_name`, "
        "`title` with a query prompt, placeholder count mismatch, or a file that can't be decoded. "
        "Batches also: too many or no items, duplicate filenames, or a missing or unused file. "
        "Batch errors name the failing item, e.g. `items[3]: ...`.",
    },
    413: {
        "model": ErrorResponse,
        "description": "Over the 8,192-token budget (per request, or per item in a batch).",
    },
    415: {"model": ErrorResponse, "description": "A file's extension isn't a supported image, audio or video type."},
    503: {"model": ErrorResponse, "description": "The GPU ran out of memory. Try a smaller request."},
}


def format_text(text: str | None, prompt_name: str | None, title: str | None) -> str | None:
    """Apply the model card's text formatting.

    Queries get a task prefix (e.g. "task: search result | query: ..."); documents get
    "title: {title} | text: ..." ("title: none" via prompt_name="Document"). Prefixes are
    applied here rather than by sentence-transformers because it would otherwise add the
    prefix as text even to media-only inputs, which the model card says not to do.
    """
    if prompt_name is not None and prompt_name not in model.prompts:
        raise HTTPException(400, f"Unknown prompt_name; choose from {sorted(model.prompts)}")
    if title is not None:
        if prompt_name is not None and not model.prompts[prompt_name].startswith("title:"):
            raise HTTPException(
                400, f"title formats the text as a document, but prompt_name={prompt_name!r} is a query prompt"
            )
        return f"title: {title} | text: {text or ''}"
    if not text:
        return None
    return (model.prompts[prompt_name] if prompt_name else "") + text


def detect_modality(name: str) -> str:
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    if ext not in EXTENSION_MODALITIES:
        raise HTTPException(
            415,
            f"Can't tell whether {name!r} is an image, audio, or video file from its extension. "
            f"Supported extensions: {', '.join(sorted(EXTENSION_MODALITIES))}",
        )
    return EXTENSION_MODALITIES[ext]


def load_media(upload: UploadFile, modality: str, stack: ExitStack):
    """Turn a file into model input.

    Returns (input, cost): estimated tokens for images and audio, and for video the number
    of frames 1 fps sampling would produce (tokens depend on how many frames fit the budget).
    """
    name = upload.filename
    upload.file.seek(0)  # the same upload may be used by several batch items
    try:
        if modality == "image":
            # Phones often store photos unrotated plus an EXIF orientation tag; apply it so the
            # model sees the photo the right way up.
            image = ImageOps.exif_transpose(Image.open(upload.file))
            return image.convert("RGB"), IMAGE_TOKENS
        if modality == "audio":
            # The model expects 16 kHz mono audio. torchcodec decodes via ffmpeg, so it
            # handles containers libsndfile can't (m4a, aac, ...), and resamples/downmixes.
            decoder = AudioDecoder(upload.file.read(), sample_rate=AUDIO_SAMPLE_RATE, num_channels=1)
            waveform = decoder.get_all_samples().data[0].numpy()
            tokens = math.ceil(len(waveform) / AUDIO_SAMPLE_RATE * AUDIO_TOKENS_PER_SECOND)
            return {"array": waveform, "sampling_rate": AUDIO_SAMPLE_RATE}, tokens
        # Video decoding works from a path, so copy the upload (in chunks, not into
        # memory) to a temp file that lives until the request's ExitStack closes.
        tmp = stack.enter_context(tempfile.NamedTemporaryFile(suffix=os.path.splitext(name)[1]))
        shutil.copyfileobj(upload.file, tmp)
        tmp.flush()
        with av.open(tmp.name) as container:
            seconds = (container.duration or 0) / av.time_base
        # Matches the video processor: int(duration * fps) frames at 1 fps, at least one.
        return tmp.name, max(1, int(seconds))
    except Exception as e:
        raise HTTPException(400, f"Could not read {modality} file {name!r}: {e}") from e


def count_text_tokens(text: str) -> int:
    return len(model.tokenizer(text, add_special_tokens=False)["input_ids"])


def budget_error(token_counts: dict[str, int], detail: str = "") -> HTTPException:
    total = sum(token_counts.values())
    return HTTPException(
        413,
        {
            "message": f"Request needs about {total} tokens; the limit is {MAX_TOKENS}. {detail}"
            "Send fewer or shorter inputs.",
            "estimated_tokens": token_counts,
        },
    )


def video_frame_cap(frames_wanted: list[int], token_budget: int) -> int:
    """Largest per-video frame cap whose total fits the token budget.

    Videos get 1 frame per second up to the cap; longer videos are sampled down to the cap
    with frames spread evenly across their whole length. One cap applies to all videos in
    the request (the processor takes it per call), so short clips keep full 1 fps coverage
    and the leftover budget goes to the longer ones.
    """
    frame_budget = token_budget // VIDEO_TOKENS_PER_FRAME
    cap = max(frames_wanted)
    while cap > 0 and sum(min(f, cap) for f in frames_wanted) > frame_budget:
        cap -= 1
    return cap


def check_placeholders(text: str, counts: dict[str, int]) -> None:
    # Placeholders are optional, but once any are used, the model's template stops
    # auto-inserting media, so every file needs exactly one placeholder.
    placeholder_counts = {m: text.count(f"<|{m}|>") for m in MEDIA_MODALITIES}
    if not any(placeholder_counts.values()):
        return
    for m in MEDIA_MODALITIES:
        if placeholder_counts[m] != counts[m]:
            raise HTTPException(
                400,
                f"text has {placeholder_counts[m]} <|{m}|> placeholder(s) but {counts[m]} {m} file(s) were "
                "sent. When using placeholders, include exactly one per file.",
            )


@dataclass
class Prepared:
    sample: dict
    modality: str
    tokens: int
    video_cap: int | None  # frames per video; None when there's no video


def prepare(text: str | None, uploads: dict[str, list[UploadFile]], stack: ExitStack) -> Prepared:
    """Validate and budget one input. `text` is already formatted."""
    if not text and not any(uploads.values()):
        raise HTTPException(400, "Provide text and/or at least one image, audio, or video file")
    if text:
        check_placeholders(text, {m: len(uploads.get(m, [])) for m in MEDIA_MODALITIES})

    sample = {"text": text} if text else {}
    token_counts = {"text": count_text_tokens(text)} if text else {}
    video_frames = []
    for m, files in uploads.items():
        if not files:
            continue
        loaded = [load_media(f, m, stack) for f in files]
        items = [item for item, _ in loaded]
        sample[m] = items[0] if len(items) == 1 else items
        if m == "video":
            video_frames = [frames for _, frames in loaded]
        else:
            token_counts[m] = sum(tokens for _, tokens in loaded)

    cap = None
    if video_frames:
        # Video gets whatever budget the other inputs leave over.
        cap = video_frame_cap(video_frames, MAX_TOKENS - sum(token_counts.values()))
        if cap == 0:
            over_without_video = sum(token_counts.values()) > MAX_TOKENS
            token_counts["video"] = len(video_frames) * VIDEO_TOKENS_PER_FRAME
            if over_without_video:
                raise budget_error(token_counts)
            raise budget_error(token_counts, "There is no room left for even one frame per video. ")
        token_counts["video"] = sum(min(f, cap) for f in video_frames) * VIDEO_TOKENS_PER_FRAME
    if sum(token_counts.values()) > MAX_TOKENS:
        raise budget_error(token_counts)
    return Prepared(sample, "+".join(sorted(sample)), sum(token_counts.values()), cap)


def encode_all(prepared: list[Prepared], dimensions: int, normalize: bool) -> list[list[float]]:
    """Embed prepared inputs in GPU batches, returning vectors in input order.

    The video frame cap is a per-call processor setting, so items are grouped by it first.
    Within a group, items are sorted by size and packed so that (items x longest item)
    stays under BATCH_TOKENS, which bounds padding and GPU memory.
    """
    results: list[list[float] | None] = [None] * len(prepared)
    groups: dict[int | None, list[int]] = {}
    for i, p in enumerate(prepared):
        groups.setdefault(p.video_cap, []).append(i)

    chunks = []
    for cap, indices in groups.items():
        indices.sort(key=lambda i: prepared[i].tokens)
        chunk = []
        for i in indices:
            # Sorted ascending, so the item being added is the longest in the chunk.
            if chunk and (len(chunk) + 1) * prepared[i].tokens > BATCH_TOKENS:
                chunks.append((cap, chunk))
                chunk = []
            chunk.append(i)
        chunks.append((cap, chunk))

    with GPU_LOCK:
        try:
            for cap, chunk in chunks:
                processing = {"audio": {"max_length": AUDIO_MAX_SAMPLES}}
                if cap is not None:
                    processing["video"] = {"max_frames": cap}
                try:
                    embs = model.encode(
                        [prepared[i].sample for i in chunk],
                        batch_size=len(chunk),
                        truncate_dim=dimensions,
                        # Applied after truncate_dim, so shortened vectors come back unit length too.
                        normalize_embeddings=normalize,
                        processing_kwargs=processing,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    raise HTTPException(503, "GPU ran out of memory; try a smaller request") from e
                except ValueError as e:
                    raise HTTPException(400, f"Could not embed input: {e}") from e
                for i, emb in zip(chunk, embs):
                    results[i] = emb.tolist()
        finally:
            trim_gpu_cache()
    return results


def trim_gpu_cache() -> None:
    """Return PyTorch's cached GPU memory to the driver once it exceeds IDLE_CACHE_MB.

    PyTorch keeps freed memory reserved for reuse, so one large request would otherwise hold
    several GB while idle. Releasing only above a threshold lets small requests keep reusing
    their cache, avoiding the cost of reallocating from the driver on every request.
    """
    if not DEVICE.startswith("cuda"):
        return
    cached = torch.cuda.memory_reserved(DEVICE) - torch.cuda.memory_allocated(DEVICE)
    if cached > IDLE_CACHE_MB * 2**20:
        torch.cuda.empty_cache()


def parse_options(model_cls, request: str | UploadFile | None):
    # Clients may send `request` as a plain text field or as a JSON-typed part with a filename
    # (Swagger UI and `curl -F 'request=...;type=application/json'` do the latter), which
    # arrives as an upload.
    if isinstance(request, StarletteUploadFile):
        try:
            request = request.file.read().decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(400, "The `request` part must be UTF-8 JSON") from e
    try:
        return model_cls.model_validate_json(request) if request else model_cls()
    except ValidationError as e:
        # e.g. "dimensions: Input should be 768, 512, 256 or 128; items.0.txt: Extra inputs are not permitted"
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" if err["loc"] else err["msg"] for err in e.errors()
        )
        raise HTTPException(400, f"Invalid `request` JSON: {problems}") from e


def group_uploads(files: list[UploadFile]) -> dict[str, list[UploadFile]]:
    uploads = {m: [] for m in MEDIA_MODALITIES}
    for f in files:
        uploads[detect_modality(f.filename or "")].append(f)
    return uploads


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model: str = Field(examples=["google/embeddinggemma-2"])
    device: str = Field(examples=["cuda"])
    dtype: str = Field(examples=["torch.float32"])


@app.get("/health", response_model=HealthResponse, summary="Check the server is up and the model is loaded")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "dtype": str(DTYPE)}


@app.post(
    "/embed",
    response_model=EmbedResponse,
    responses=ERROR_RESPONSES,
    summary="Embed text and files into one vector",
    description="Combines everything in the request (text plus any number of images, audio clips and videos) "
    "into **one** vector. Needs text or at least one file.\n\n"
    "```\ncurl host:11433/embed -F 'request={\"text\": \"see <|image|>\", \"prompt_name\": \"SearchQuery\"}' "
    "-F file=@photo.jpg\n```",
)
def embed(
    request: str | UploadFile | None = Form(None, description="JSON options; see the EmbedOptions schema. Optional."),
    file: list[UploadFile] = File([], description="Files to embed, one part each. Type comes from the extension."),
):
    opts = parse_options(EmbedOptions, request)
    text = format_text(opts.text, opts.prompt_name, opts.title)
    with ExitStack() as stack:
        item = prepare(text, group_uploads(file), stack)
        [embedding] = encode_all([item], opts.dimensions, opts.normalize)
    return {"modality": item.modality, "dimensions": opts.dimensions, "embedding": embedding}


@app.post(
    "/embed/batch",
    response_model=BatchResponse,
    responses=ERROR_RESPONSES,
    summary="Embed many inputs at once, one vector each",
    description="Embeds each item in `items` separately and returns one vector per item, in order. Faster than "
    "separate requests, especially for text. Each item follows the same rules and token budget as `/embed`; "
    "if any item is invalid, the whole batch fails and the error names the item.\n\n"
    "Files are uploaded as `file` parts and referenced from items **by filename**. Filenames must be unique, "
    "every uploaded file must be used, and one file can be shared by several items.",
)
def embed_batch(
    request: str | UploadFile = Form(..., description="JSON options; see the BatchOptions schema. Required."),
    file: list[UploadFile] = File([], description="Files referenced by the items' `files` lists, by filename."),
):
    opts = parse_options(BatchOptions, request)
    if not 1 <= len(opts.items) <= MAX_BATCH_ITEMS:
        raise HTTPException(400, f"items must contain between 1 and {MAX_BATCH_ITEMS} entries")

    by_name: dict[str, UploadFile] = {}
    for f in file:
        if f.filename in by_name:
            raise HTTPException(400, f"Two uploaded files are named {f.filename!r}; filenames must be unique")
        by_name[f.filename] = f
    referenced = {name for item in opts.items for name in item.files}
    if missing := sorted(referenced - by_name.keys()):
        raise HTTPException(400, f"items reference files that weren't uploaded: {missing}")
    if unused := sorted(by_name.keys() - referenced):
        raise HTTPException(400, f"uploaded files not used by any item: {unused}")

    with ExitStack() as stack:
        prepared = []
        for i, item in enumerate(opts.items):
            try:
                text = format_text(item.text, item.prompt_name, item.title)
                prepared.append(prepare(text, group_uploads([by_name[n] for n in item.files]), stack))
            except HTTPException as e:
                # Point at the failing item; budget errors carry a dict detail.
                detail = {**e.detail, "item": i} if isinstance(e.detail, dict) else f"items[{i}]: {e.detail}"
                raise HTTPException(e.status_code, detail) from e
        embeddings = encode_all(prepared, opts.dimensions, opts.normalize)
    return {
        "dimensions": opts.dimensions,
        "results": [{"modality": p.modality, "embedding": e} for p, e in zip(prepared, embeddings)],
    }


def custom_openapi() -> dict:
    """Describe the JSON `request` part properly in /docs.

    FastAPI only knows `request` as a form string, so the generated docs hide every option.
    Point it at the options schema instead (with encoding application/json), and list the
    model's actual prompt names as allowed `prompt_name` values.
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    components = schema["components"]["schemas"]
    for cls in (EmbedOptions, BatchOptions):
        js = cls.model_json_schema(ref_template="#/components/schemas/{model}")
        components.update(js.pop("$defs", {}))
        components[cls.__name__] = js
    if model is not None:
        for name in ("EmbedOptions", "ItemOptions"):
            for branch in components[name]["properties"]["prompt_name"].get("anyOf", []):
                if branch.get("type") == "string":
                    branch["enum"] = sorted(model.prompts)
    for path, options in (("/embed", "EmbedOptions"), ("/embed/batch", "BatchOptions")):
        body = schema["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]
        form = components[body["schema"]["$ref"].rsplit("/", 1)[1]]
        form["properties"]["request"] = {
            "$ref": f"#/components/schemas/{options}",
            "description": form["properties"]["request"].get("description", ""),
        }
        body["encoding"] = {"request": {"contentType": "application/json"}}
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
