# Backend

The Flask API and training controller are in [backend.py](backend.py). The
backend listens locally on `http://127.0.0.1:5000`; its `/api/health` endpoint
checks server readiness, and `/api/status` reports the latest training state.
Training, pause, resume, stop, speed, reset, and model upload endpoints are
implemented in that module.

Use the repository's `start_dashboard.bat` to start the backend and the
separate static frontend together. Training additionally requires the optional
simulation dependencies described in the root [README](../README.md).
