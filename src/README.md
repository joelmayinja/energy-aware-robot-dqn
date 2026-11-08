# Backend

The Flask application, training API, and training controller are in
[backend.py](backend.py). It serves the dashboard page at `/`, frontend assets
at `/assets/`, and its API from the same origin. `/api/health` checks server
readiness, and `/api/status` reports training state. Training, pause, resume,
stop, speed, reset, and model upload endpoints are implemented in that module.

Use the repository's `start_dashboard.bat` on Windows to launch the complete
application with one command. In Codespaces, the existing
`src.dashboard:app` Flask import remains available and points to this same app.
Training additionally requires the optional simulation dependencies described
in the root [README](../README.md).
