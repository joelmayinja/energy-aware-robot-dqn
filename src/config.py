"""Shared local dashboard address."""

HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"
FRONTEND_PORT = 8000
FRONTEND_URL = f"http://{HOST}:{FRONTEND_PORT}"
FRONTEND_ORIGINS = {
    FRONTEND_URL,
    f"http://localhost:{FRONTEND_PORT}",
}
