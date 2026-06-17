"""
Entry point so the API starts with ``python -m src.api``.

Host/port are overridable via the API_HOST / API_PORT env vars. A single worker
is used on purpose: the models are loaded into this process at startup, so extra
workers would each reload them and multiply the memory footprint.
"""

import os


def main():
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("src.api.app:create_app", factory=True, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
