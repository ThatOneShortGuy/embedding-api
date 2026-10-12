# Start the embedding server (settings come from .env or the environment; see .env.example)
start:
    .venv/bin/python app.py

# Create the virtualenv and install dependencies
setup:
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

# Run as the user the service should run as; it fills in that user and this directory.
# Install and enable the systemd service so the server starts on boot (asks for sudo)
install-service:
    @if [ "$(id -u)" -eq 0 ]; then echo "Run as the user the service should run as, not root or sudo"; exit 1; fi
    sed -e "s|@USER@|$(id -un)|g" -e "s|@DIR@|{{justfile_directory()}}|g" embedding-api.service \
        | sudo tee /etc/systemd/system/embedding-api.service > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now embedding-api

# Follow the service logs from the systemd journal (pass extra journalctl flags, e.g. `just logs --since today`)
logs *ARGS:
    journalctl -u embedding-api -f {{ARGS}}
