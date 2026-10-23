# EmbeddingGemma API

A small HTTP API around [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2).
Send text, images, audio, video, or any mix of them, and get back one embedding vector.

```bash
curl localhost:11433/embed \
  -F 'request={"text": "watch <|video|> see <|image|>"}' \
  -F file=@demo.mp4 -F file=@photo.jpg
```
```json
{"modality": "image+text+video", "dimensions": 768, "embedding": [0.0123, -0.0456, ...]}
```

## Setup

Requirements:
- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- FFmpeg shared libraries, which torchcodec needs to decode audio and video (`apt install ffmpeg`)
- An NVIDIA GPU. It also runs on CPU (`DEVICE=cpu`) with identical results, but far slower for images and
  video; see [Running on CPU](#running-on-cpu).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
```

`.env` holds your local settings (see [Configuration](#configuration)) and is ignored by git. The defaults
work as-is.

The `torch` wheel from PyPI already includes CUDA. To check that it can see the GPU:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The model (~1.4 GB) downloads from Hugging Face the first time the server starts. No login is needed.

## Running

```bash
.venv/bin/python app.py
```

With the default settings this serves on `0.0.0.0:11433`, so it's reachable from other machines on the network.
`GET /health` reports the model, device and precision once the model has loaded.

To run it as a systemd service that starts on boot, run this as the user it should run as (not with sudo):

```bash
just install-service
```

`embedding-api.service` is a template: the recipe fills in your user and this project's directory, installs the
result to `/etc/systemd/system/`, and enables it. Rerun it after moving the project or editing the template.
Follow the logs with `just logs`.

Interactive API docs are at `http://<host>:11433/docs`. They list every option with its type, default and
allowed values (including all `prompt_name` values), the response format and the error codes.

## API

### `POST /embed`

The request is `multipart/form-data` with two kinds of parts:

| Part | | |
|---|---|---|
| `request` | optional | JSON with the options below |
| `file` | optional, repeatable | Raw file bytes. The filename's extension decides whether it's an image, audio or video. |

Each request needs text or at least one file. All inputs in one request are combined into a **single** vector.
Unknown or misspelled option names in `request` are rejected with `400`, not ignored.

Options in `request`:

| Field | Default | |
|---|---|---|
| `text` | `null` | Text to embed. Can contain media placeholders (see below). |
| `prompt_name` | `null` | Task prefix for the text, e.g. `"SearchQuery"`, `"Document"`. See [Text formatting](#text-formatting). |
| `title` | `null` | Formats the text as a titled document: `title: {title} \| text: {text}` |
| `dimensions` | `768` | `768`, `512`, `256` or `128`. Shorter vectors are the first N values of the full vector. |
| `normalize` | `false` | `true` scales the vector to unit length, after shortening it to `dimensions`. |

Response:

| Field | |
|---|---|
| `modality` | The input types used, e.g. `"text"`, `"image"`, `"audio+image+text+video"` |
| `dimensions` | Length of `embedding` |
| `embedding` | The vector: raw by default, unit length with `"normalize": true` (see below) |

By default, vectors are returned raw, as the model's mean pooling produces them. Their lengths vary
widely (roughly 200 to 1,300). Keep in mind:

- **Use cosine similarity, not a plain dot product.** A dot product on raw vectors is dominated by vector
  length.
- **Normalize after shortening.** The model card says a vector cut to 512, 256 or 128 dimensions must be
  re-normalized before comparing; skipping it silently degrades ranking. Cosine similarity does this for you.
- **If your vector database only supports dot product** or inner product, send `"normalize": true`. Unit-length
  vectors make the dot product equal to cosine similarity, and normalization happens after shortening, as the
  model card requires.

A normalized vector is exactly the raw vector divided by its length, so the direction is the same either way.

### Examples

Text only:
```bash
curl localhost:11433/embed -F 'request={"text": "waterproof trail shoes", "prompt_name": "SearchQuery"}'
```

A file on its own (no `request` part needed):
```bash
curl localhost:11433/embed -F file=@clip.m4a
```

Everything at once, at 256 dimensions:
```bash
curl localhost:11433/embed \
  -F 'request={"text": "Product demo", "dimensions": 256}' \
  -F file=@shoe.jpg -F file=@narration.mp3 -F file=@demo.mp4
```

Python:
```python
import json, requests

resp = requests.post(
    "http://localhost:11433/embed",
    data={"request": json.dumps({"text": "Upper: <|image|> Sole: <|image|>", "title": "Trail shoe"})},
    files=[("file", open("upper.jpg", "rb")), ("file", open("sole.jpg", "rb"))],
)
vector = resp.json()["embedding"]
```

### `POST /embed/batch`

Embeds many independent inputs in one request and returns one vector per item, in order. Use it for
throughput, especially with text (see [Performance](#performance)).

The request is `multipart/form-data`:

| Part | | |
|---|---|---|
| `request` | required | JSON: `{"items": [...], "dimensions": 768}` |
| `file` | optional, repeatable | Raw file bytes, referenced from items **by filename** |

Each item takes the same options as `/embed` (`text`, `prompt_name`, `title`), plus `files`: the filenames of
uploaded parts that belong to that item. `dimensions` and `normalize` apply to the whole batch.

- Filenames must be unique within a request.
- One uploaded file can be used by several items, e.g. the same image paired with different texts.
- Every uploaded file must be used by at least one item, and every name an item lists must be uploaded.

```bash
curl localhost:11433/embed/batch \
  -F 'request={"items": [
        {"text": "waterproof trail shoes", "prompt_name": "SearchQuery"},
        {"title": "Trail shoe", "text": "Grippy sole <|image|>", "files": ["shoe.jpg"]},
        {"files": ["demo.mp4"]},
        {"text": "same photo, different caption", "files": ["shoe.jpg"]}
      ]}' \
  -F file=@shoe.jpg -F file=@demo.mp4
```

Response:

```json
{"dimensions": 768,
 "results": [{"modality": "text", "embedding": [...]},
             {"modality": "image+text", "embedding": [...]},
             {"modality": "video", "embedding": [...]},
             {"modality": "image+text", "embedding": [...]}]}
```

Python:
```python
import json, requests

texts = ["first document", "second document", "third document"]
resp = requests.post(
    "http://localhost:11433/embed/batch",
    data={"request": json.dumps({"items": [{"text": t, "prompt_name": "Document"} for t in texts]})},
)
vectors = [r["embedding"] for r in resp.json()["results"]]
```

Each item gets exactly the same vector it would get from `/embed`, and follows the same rules: its own
8,192-token budget, placeholders, text formatting and video sampling. Everything is checked before anything
runs, so if any item is invalid the whole batch fails and nothing is embedded. Error messages say which item
failed (`items[3]: ...`, or an `"item": 3` field on token-budget errors).

A batch can hold up to 256 items. The server splits it into GPU batches of at most 16,384 tokens, counting
each item as long as the longest item in its GPU batch, which keeps memory use bounded. A batch of many large
items is therefore processed a few items at a time; it still works, just without much speedup.

## Supported files

The type comes from the file extension (case doesn't matter):

| Type | Extensions |
|---|---|
| Image | jpg, jpeg, jfif, jpe, pjpeg, png, gif, webp, bmp, tif, tiff, avif, heic, heif |
| Audio | wav, mp3, m4a, aac, flac, ogg, oga, opus |
| Video | mp4, m4v, mov, avi, mkv, webm, mpeg, mpg, 3gp, ogv, wmv, flv, ts |

Files with another extension, or none, are rejected with `415`.
Audio is converted to 16 kHz mono automatically, as the model requires.

Photos are turned the right way up using their EXIF orientation tag, which phones often use instead of
rotating the pixels, so a portrait photo isn't embedded sideways.

HEIC/HEIF support comes from the `pillow-heif` package. Its prebuilt wheels bundle libheif and libde265
(LGPL) and the x265 encoder (GPL), and HEIC's HEVC compression is patent-encumbered. That's usually fine for
internal use; check it before shipping this in a product.

## Combining inputs

Without placeholders, media is added after the text in this order: images, then audio, then video,
regardless of upload order.

To control where each item goes, put `<|image|>`, `<|audio|>` or `<|video|>` in the text.
Each placeholder takes the next uploaded file of that type, in upload order:

```bash
curl localhost:11433/embed \
  -F 'request={"text": "Waterproof running shoes. <|image|> Breathable mesh upper. <|image|> Grip test: <|video|>"}' \
  -F file=@shoe.jpg -F file=@mesh.jpg -F file=@demo.mp4
```

Placeholders are all-or-nothing. If the text has any, it must have exactly one for every file sent.
That's how the model handles them, and a mismatch returns `400`.

## Text formatting

The model was trained with short prefixes on text. Using the right one improves quality. Prefixes only
apply to text: media never gets one, even if `prompt_name` is set on a media-only request.

For search, use a query prefix on queries and the document format on the things being searched:

| Use | Queries | Documents |
|---|---|---|
| Search | `prompt_name: "SearchQuery"` | `title: "..."`, or `prompt_name: "Document"` if there's no title |
| Question answering | `"QuestionAnswering"` | same as above |
| Fact checking | `"FactChecking"` | same as above |
| Code search | `"CodeRetrieval"` | same as above, with a filename as the title |

For comparing like with like, use the same prompt on everything: `"Classification"`, `"Clustering"`,
or `"SentenceSimilarity"`.

What the model receives:

| Request | Text sent to the model |
|---|---|
| `{"text": "aurora", "prompt_name": "SearchQuery"}` | `task: search result \| query: aurora` |
| `{"text": "...", "prompt_name": "Document"}` | `title: none \| text: ...` |
| `{"text": "...", "title": "Aurora"}` | `title: Aurora \| text: ...` |

`title` can't be combined with a query prompt such as `SearchQuery`; that returns `400`.
An unknown `prompt_name` returns `400` with the list of valid names.

## Limits

Everything in one request shares the model's **8,192-token** context. Requests estimated to need more are
rejected with `413` and a breakdown by type:

```json
{"detail": {"message": "Request needs about 10080 tokens; the limit is 8192. Send fewer or shorter inputs.",
            "estimated_tokens": {"image": 5600, "video": 4480}}}
```

| Input | Cost | Most that fits alone |
|---|---|---|
| Text | ~1 token per word piece | ~8,000 tokens |
| Image | 280 tokens each | 29 images |
| Audio | 25 tokens per second | ~327 seconds |
| Video | 140 tokens per frame | 58 frames |

**Video** is sampled at 1 frame per second, using whatever budget text, images and audio leave over.
When a video has more seconds than fit, it gets as many frames as fit, spread evenly across its whole length,
so the end isn't cut off. For example, a 60-second video alone gets 58 frames, and with 10 images it gets 38.
When several videos share a request, short clips keep 1 frame per second and the long ones share what's left.

**Audio** uses the full length up to the budget. The model's audio processor normally keeps only the first
30 seconds and drops the rest without any error; this server lifts that limit.

## Configuration

Settings are read from environment variables, or from a `.env` file next to `app.py`. `.env.example` lists
every setting with its default and an explanation; copy it to `.env` and edit. Variables already set in the
environment override `.env`, and an empty value means "use the default".

| Variable | Default | |
|---|---|---|
| `HOST` | `0.0.0.0` | Address to listen on. `127.0.0.1` for local-only. |
| `PORT` | `11433` | Port |
| `MODEL_ID` | `google/embeddinggemma-2` | |
| `DEVICE` | `cuda` if available, else `cpu` | |
| `DTYPE` | `float32` | `float32` or `bfloat16`. Use `bfloat16` on GPUs that support it (Ampere or newer) to halve memory. `float16` is refused at startup: this model silently produces NaN or degraded vectors in it. |
| `MAX_TOKENS` | `8192` | Token budget per input (per request for `/embed`, per item for `/embed/batch`). Raising it lets inputs run past what the model was trained on, and uses more GPU memory. |
| `BATCH_TOKENS` | `16384` | Most tokens in one GPU batch (items × longest item). ~8 GB of GPU memory for images at the default. Lower it on smaller GPUs. |
| `MAX_BATCH_ITEMS` | `256` | Most items in one `/embed/batch` request |
| `IDLE_CACHE_MB` | `1024` | Most cached, unused GPU memory kept after a request. Above it, the cache is given back to the GPU. See [Notes](#notes). |
| `HF_TOKEN`, `HF_HOME` | unset | Optional Hugging Face download token and model cache location |

Write `.env` as plain `KEY=value` lines with comments on their own lines: the systemd service reads the same
file, and systemd doesn't support comments after a value.

## Errors

| Status | When |
|---|---|
| `400` | Nothing to embed, invalid `request` JSON, unknown `prompt_name`, bad `dimensions`, placeholder count mismatch, or a file that can't be decoded. For batches also: too many or no items, duplicate filenames, a missing file, or an unused file. |
| `413` | Over the token budget |
| `415` | File extension not supported |
| `422` | `/embed/batch` called without a `request` part |
| `503` | GPU ran out of memory |

The `detail` field explains what went wrong.

## Notes

- **No authentication.** Anyone who can reach the port can use the API, and your GPU. Keep it on a trusted
  network, or put it behind a firewall or a reverse proxy that checks access.
- **No upload size limit.** Uploads are written to temporary files on disk instead of held in memory,
  but very large files still use disk space and time to decode.
- **One request uses the GPU at a time.** Concurrent requests are accepted, but they queue for the GPU
  instead of running together, so several large requests at once don't run out of memory. To get more
  throughput, send batches, not parallel requests.
- **GPU memory is given back after large requests.** PyTorch keeps memory it has used reserved for reuse.
  After each request, if more than `IDLE_CACHE_MB` (1 GB by default) is cached but unused, the server returns
  it to the GPU, so idle usage settles near the model's ~3.3 GB instead of staying at ~18 GB after a large
  batch. Small requests stay under the threshold and keep reusing their cache. Peak usage during a request is
  unchanged: a full batch still needs ~8 GB on top of the model, and fails with `503` if another program has
  taken that memory.

## Performance

Measured on a TITAN RTX in float32, over HTTP on the local network:

| Input | `/embed`, one per request | `/embed/batch` | Speedup |
|---|---|---|---|
| Short texts (5–60 words) | 18 per second | 235 per second | 13× |
| Images | 5 per second | 7 per second | 1.3× |
| Audio clips (3–12 s) | 8.5 per second | 18 per second | 2× |

A single request is fast: text, images and audio return well under a second. A request filling the token
budget (29 images, or a long video) takes about 5 seconds.

Images gain little from batching because the GPU is already fully busy: each image is 2,520 patches through
the vision encoder, and almost all the time is spent in the encoder's matrix maths. Faster number formats
don't help on this card either: it has no native bfloat16, and float16 breaks this model. On a GPU with
bfloat16 support (Ampere or newer), `DTYPE=bfloat16` should be noticeably faster, but that's untested here.

### Running on CPU

With `DEVICE=cpu` everything works and the vectors match the GPU's (cosine similarity 1.000000 for every input
type). It's practical for text, but slow for media. Measured on a 16-core CPU, float32, one request at a time:

| Input | CPU | GPU (TITAN RTX) |
|---|---|---|
| Short text | ~0.1 s | ~0.05 s |
| 64 texts in one batch | 1.7 s (38 per second) | 0.08 s (845 per second) |
| One image | ~7 s | ~0.25 s |
| 3-second audio clip | 1.5 s | 0.3 s |
| 60-second audio clip | 10 s | 0.4 s |
| 10-second video (10 frames) | 33 s | 0.8 s |
| 60-second video (58 frames) | ~3.5 minutes | 4.5 s |

Requests are handled one at a time, so a long video blocks everything behind it for minutes. Make sure clients'
HTTP timeouts allow for that. The CPU server uses about 4.3 GB of RAM. Keep `DTYPE=float32` on CPU.
