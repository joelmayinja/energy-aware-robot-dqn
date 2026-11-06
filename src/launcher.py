"""Prepare the lightweight dashboard environment and launch both local services."""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import subprocess
import sys
from threading import Thread
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FRONTEND_PORT, FRONTEND_URL, HOST, PORT, URL


CONDA_PYTHON = PROJECT_ROOT / ".conda-training" / "python.exe"
VENV_PYTHON = PROJECT_ROOT / ".venv310" / "Scripts" / "python.exe"
FRONTEND_ROOT = PROJECT_ROOT / "web"


def _runtime_python():
    return CONDA_PYTHON if CONDA_PYTHON.is_file() else VENV_PYTHON


def _port_is_occupied(port):
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def _backend_at_port():
    """Return whether the API port serves this backend, is occupied, or is free."""
    try:
        with urlopen(f"{URL}/api/health", timeout=1) as response:
            data = json.loads(response.read())
        return (
            "dashboard"
            if isinstance(data, dict)
            and data.get("application") == "energy-consumption-dqn"
            else "occupied"
        )
    except HTTPError as error:
        if error.code == 404:
            try:
                with urlopen(f"{URL}/api/status", timeout=1) as response:
                    status = json.loads(response.read())
                if isinstance(status, dict) and {
                    "pybullet_available",
                    "run_id",
                    "episodes_requested",
                } <= status.keys():
                    return "legacy_dashboard"
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
                pass
        return "occupied"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return "occupied" if _port_is_occupied(PORT) else "free"


def _frontend_at_port():
    """Return whether the frontend port serves the project page or is occupied."""
    try:
        with urlopen(FRONTEND_URL, timeout=1) as response:
            page = response.read(4096).decode("utf-8", errors="replace")
        return (
            "frontend"
            if "<title>Wallie | Energy Intelligence</title>" in page
            else "occupied"
        )
    except (HTTPError, URLError, TimeoutError, OSError):
        return "occupied" if _port_is_occupied(FRONTEND_PORT) else "free"


def _ensure_runtime_environment():
    using_conda_environment = CONDA_PYTHON.is_file()
    if not using_conda_environment and sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Python 3.10 is required for the default lightweight environment. "
            "Install it, or create the optional .conda-training environment described in README.md."
        )
    if not using_conda_environment and not VENV_PYTHON.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_PYTHON.parent.parent)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    runtime_python = _runtime_python()
    version = subprocess.run(
        [str(runtime_python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if version != "3.10":
        raise RuntimeError(
            f"The dashboard environment uses Python {version}, not Python 3.10. "
            "Recreate it with Python 3.10 before continuing."
        )

    flask_check = subprocess.run(
        [str(runtime_python), "-c", "import importlib.util; raise SystemExit(importlib.util.find_spec('flask') is None)"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if flask_check.returncode != 0:
        print("Installing the lightweight dashboard dependency (Flask)...")
        subprocess.run(
            [str(runtime_python), "-m", "pip", "install", "-r", "requirements-dashboard.txt"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return runtime_python


def _wait_for_service(threads, service_check, description):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not all(thread.is_alive() for thread in threads):
            raise RuntimeError(f"A server stopped while starting the {description}.")
        if service_check():
            return
        time.sleep(0.25)
    raise RuntimeError(f"{description} did not become ready within 30 seconds.")


def main():
    backend_state = _backend_at_port()
    frontend_state = _frontend_at_port()
    if backend_state == "legacy_dashboard":
        print(
            f"An older combined dashboard is running at {URL}. "
            "Stop its server window with Ctrl+C, then run start_dashboard.bat again."
        )
        return 2
    if backend_state == "occupied":
        print(
            f"Port {PORT} is being used by another application. "
            "Close it and retry; the backend uses this fixed port."
        )
        return 2
    if frontend_state == "occupied":
        print(
            f"Port {FRONTEND_PORT} is being used by another application. "
            "Close it and retry; the frontend uses this fixed port."
        )
        return 2
    if backend_state == "dashboard" and frontend_state == "frontend":
        print(f"Dashboard is already running at {FRONTEND_URL}; opening it in your browser.")
        webbrowser.open(FRONTEND_URL)
        return 0

    try:
        runtime_python = _ensure_runtime_environment()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"Dashboard setup failed: {error}")
        return 1

    if Path(sys.executable).resolve() != runtime_python.resolve():
        try:
            return subprocess.run(
                [str(runtime_python), str(Path(__file__).resolve())],
                cwd=PROJECT_ROOT,
                check=False,
            ).returncode
        except KeyboardInterrupt:
            return 0

    servers = []
    threads = []
    try:
        from werkzeug.serving import make_server

        from src.dashboard import app

        if backend_state == "free":
            servers.append(make_server(HOST, PORT, app, threaded=True))

        if frontend_state == "free":
            frontend_handler = partial(
                SimpleHTTPRequestHandler, directory=str(FRONTEND_ROOT)
            )
            frontend = ThreadingHTTPServer(
                (HOST, FRONTEND_PORT), frontend_handler
            )
            frontend.daemon_threads = True
            servers.append(frontend)

        threads = [
            Thread(
                target=server.serve_forever,
                name=f"dashboard-server-{index}",
                daemon=True,
            )
            for index, server in enumerate(servers)
        ]
        if backend_state == "free":
            print(f"Starting the backend API at {URL}...")
        if frontend_state == "free":
            print(f"Starting the frontend at {FRONTEND_URL}...")
        for thread in threads:
            thread.start()

        _wait_for_service(
            threads,
            lambda: backend_state != "free" or _backend_at_port() == "dashboard",
            "backend API",
        )
        _wait_for_service(
            threads,
            lambda: frontend_state != "free" or _frontend_at_port() == "frontend",
            "frontend",
        )

        print(
            f"Frontend: {FRONTEND_URL}\n"
            f"Backend API: {URL}\n"
            "Keep this window open; press Ctrl+C to stop both services."
        )
        webbrowser.open(FRONTEND_URL)
        while all(thread.is_alive() for thread in threads):
            time.sleep(0.5)
        print("A dashboard server stopped unexpectedly.")
        return 1
    except KeyboardInterrupt:
        print("\nStopping the frontend and backend...")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"Dashboard startup failed: {error}")
        return 1
    finally:
        for server, thread in reversed(list(zip(servers, threads))):
            if thread.is_alive():
                server.shutdown()
            server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
