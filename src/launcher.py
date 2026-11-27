"""Prepare the dashboard environment and launch the combined web application."""

import json
from pathlib import Path
import socket
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HOST, PORT, URL


CONDA_PYTHON = PROJECT_ROOT / ".conda-training" / "python.exe"
VENV_PYTHON = PROJECT_ROOT / ".venv310" / "Scripts" / "python.exe"


def _runtime_python():
    return CONDA_PYTHON if CONDA_PYTHON.is_file() else VENV_PYTHON


def _port_is_occupied(port):
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def _backend_at_port():
    """Return whether the application API is available, the port is occupied, or free."""
    try:
        with urlopen(f"{URL}/api/health", timeout=1) as response:
            data = json.loads(response.read())
        if not isinstance(data, dict) or data.get("application") != "energy-consumption-dqn":
            return "occupied"
        with urlopen(URL, timeout=1) as response:
            page = response.read(4096).decode("utf-8", errors="replace")
        is_current_dashboard = (
            "<title>Wallie | Energy Intelligence</title>" in page
            and 'name="wattwise-entry" content="demo-auth-v1"' in page
        )
        return "dashboard" if is_current_dashboard else "legacy_dashboard"
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
        [
            str(runtime_python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
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
        [
            str(runtime_python),
            "-c",
            "import importlib.util; raise SystemExit(importlib.util.find_spec('flask') is None)",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if flask_check.returncode != 0:
        print("Installing the lightweight dashboard dependency (Flask)...")
        subprocess.run(
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                "-r",
                "requirements-dashboard.txt",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return runtime_python


def main():
    application_state = _backend_at_port()
    if application_state == "legacy_dashboard":
        print(
            f"An older dashboard is running at {URL}. "
            "Stop its server window with Ctrl+C, then run start_dashboard.bat again."
        )
        return 2
    if application_state == "occupied":
        print(
            f"Port {PORT} is being used by another application. "
            "Close it and retry; the dashboard uses this fixed port."
        )
        return 2
    if application_state == "dashboard":
        print(f"Dashboard is already running at {URL}; opening it in your browser.")
        webbrowser.open(URL)
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

    try:
        from src.backend import app

        print(f"Dashboard: {URL}")
        print("Frontend and training API are served by this one application.")
        print("Stop the dashboard with Ctrl+C.")
        webbrowser.open(URL)
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nStopping the dashboard...")
        return 0
    except OSError as error:
        print(f"Dashboard startup failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
