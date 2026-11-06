"""Prepare the project's Python environment and launch the local dashboard."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HOST, PORT, URL


VENV_PYTHON = PROJECT_ROOT / ".venv310" / "Scripts" / "python.exe"
REQUIRED_MODULES = ("flask", "gymnasium", "numpy", "pybullet", "torch")


def _dashboard_at_port():
    """Return whether port 5000 serves this dashboard, is occupied, or is free."""
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
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                return "occupied"
        except OSError:
            return "free"


def _has_msvc_toolchain():
    if shutil.which("cl.exe"):
        return True
    installer = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not installer.is_file():
        return False
    result = subprocess.run(
        [
            str(installer),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _ensure_venv():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Python 3.10 is required. Install it, then run start_dashboard.bat again."
        )
    if not VENV_PYTHON.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_PYTHON.parent.parent)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    version = subprocess.run(
        [str(VENV_PYTHON), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if version != "3.10":
        raise RuntimeError(
            f".venv310 uses Python {version}, not Python 3.10. Recreate that environment before continuing."
        )

    module_check = (
        "import importlib.util; "
        f"modules = {REQUIRED_MODULES!r}; "
        "print(','.join(name for name in modules if importlib.util.find_spec(name) is None))"
    )
    missing_result = subprocess.run(
        [str(VENV_PYTHON), "-c", module_check],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    missing = missing_result.stdout.strip().split(",") if missing_result.stdout.strip() else []
    if missing:
        if "pybullet" in missing and sys.platform == "win32" and not _has_msvc_toolchain():
            raise RuntimeError(
                "PyBullet needs Microsoft C++ Build Tools on Windows. Install Visual Studio "
                "Build Tools with the 'Desktop development with C++' workload (MSVC and a "
                "Windows SDK), then run start_dashboard.bat again."
            )
        print(f"Installing project dependencies (missing: {', '.join(missing)})...")
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=PROJECT_ROOT,
            check=True,
        )


def main():
    port_state = _dashboard_at_port()
    if port_state == "dashboard":
        print(f"Dashboard is already running at {URL}; opening it in your browser.")
        webbrowser.open(URL)
        return 0
    if port_state == "legacy_dashboard":
        print(
            f"An older version of this dashboard is already running at {URL}. "
            "Stop its server window with Ctrl+C, then run start_dashboard.bat again."
        )
        return 2
    if port_state == "occupied":
        print(
            f"Port {PORT} is being used by another application. "
            "The dashboard stays on this port; close the conflicting application and retry."
        )
        return 2

    try:
        _ensure_venv()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"Dashboard setup failed: {error}")
        return 1

    port_state = _dashboard_at_port()
    if port_state == "dashboard":
        print(f"Dashboard is already running at {URL}; opening it in your browser.")
        webbrowser.open(URL)
        return 0
    if port_state == "legacy_dashboard":
        print(
            f"An older version of this dashboard is already running at {URL}. "
            "Stop its server window with Ctrl+C, then run start_dashboard.bat again."
        )
        return 2
    if port_state == "occupied":
        print(
            f"Port {PORT} is being used by another application. "
            "The dashboard stays on this port; close the conflicting application and retry."
        )
        return 2

    print(f"Starting the dashboard at {URL}. Keep this window open; press Ctrl+C to stop.")
    process = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "src.dashboard"],
        cwd=PROJECT_ROOT,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                print(f"Dashboard server exited during startup (code {process.returncode}).")
                return process.returncode or 1
            if _dashboard_at_port() == "dashboard":
                webbrowser.open(URL)
                return process.wait()
            time.sleep(0.25)
        print("Dashboard did not become ready within 30 seconds.")
        process.terminate()
        return 1
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
