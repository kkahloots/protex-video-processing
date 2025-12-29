#!/usr/bin/env python
"""
Environment setup script for Protex CV Ops Take-Home.

- Local: creates .venv and installs requirements
- Colab / Jupyter: skips venv and installs via pip

Usage:
    python setup_env.py
    # or inside Colab:
    #   !python setup_env.py
"""

import os
import sys
import subprocess
from pathlib import Path


def detect_root():
    """Detect root directory—works inside notebooks and scripts."""
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    else:
        # Notebook or interactive mode
        print("Running in a notebook / interactive environment. Using cwd.")
        return Path(os.getcwd())


ROOT = detect_root()
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"


def in_colab():
    """Return True if running inside Google Colab."""
    try:
        import google.colab

        return True
    except ImportError:
        return False


def run(cmd, **kwargs):
    print(f"--> Running: {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd, **kwargs)


def install_in_colab():
    if not REQ_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQ_FILE}")

    print("🚀 Detected Google Colab environment")
    print("Installing dependencies with pip...")
    run([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)])
    print("✅ Colab environment ready.")


def create_venv_and_install():
    if not REQ_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQ_FILE}")

    # Determine venv python/pip paths
    if os.name == "nt":
        pip_path = VENV_DIR / "Scripts" / "pip.exe"
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:
        pip_path = VENV_DIR / "bin" / "pip"
        python_path = VENV_DIR / "bin" / "python"

    if VENV_DIR.exists() and pip_path.exists():
        print(f"✅ Virtual environment already exists at {VENV_DIR}")
    else:
        print("🖥️ Creating local virtual environment in .venv ...")
        try:
            run([sys.executable, "-m", "venv", str(VENV_DIR)])
        except subprocess.CalledProcessError:
            print(
                "⚠️  venv creation failed. Installing to current environment instead..."
            )
            run([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)])
            print("✅ Dependencies installed in current environment.")
            return

    print(f"Using venv Python: {python_path}")
    print("Installing dependencies...")
    run([str(pip_path), "install", "--upgrade", "-r", str(REQ_FILE)])

    print("\n🎉 Virtual environment ready.")
    print("Activate it:")
    if os.name == "nt":
        print(r"  .venv\Scripts\activate")
    else:
        print(r"  source .venv/bin/activate")


def main():
    if in_colab():
        install_in_colab()
    else:
        create_venv_and_install()


if __name__ == "__main__":
    main()
