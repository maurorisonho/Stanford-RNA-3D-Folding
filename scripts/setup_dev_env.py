#!/usr/bin/env python3

"""
Bootstrap a local development environment for the Stanford RNA 3D Folding project.

This script creates (or reuses) a virtual environment at the repository root under
``.venv`` and installs the core dependencies listed in
``stanford_rna3d/requirements.txt``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a command, streaming output and raising on failure."""
    subprocess.run(cmd, check=True, cwd=cwd)


def venv_python(venv_dir: Path) -> Path:
    """Return the path to the Python executable inside the venv."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_pip(venv_dir: Path) -> Path:
    """Return the path to the pip executable inside the venv."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = repo_root / ".venv"
    requirements_file = repo_root / "stanford_rna3d" / "requirements.txt"

    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    if not venv_dir.exists():
        print(f"Creating virtual environment at {venv_dir}...")
        run([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        print(f"Reusing existing virtual environment at {venv_dir}.")

    python_exe = venv_python(venv_dir)
    pip_exe = venv_pip(venv_dir)

    if not pip_exe.exists():
        # Ensure pip is available (e.g., venv created with --without-pip)
        print("pip not found in virtual environment; bootstrapping pip...")
        run([str(python_exe), "-m", "ensurepip", "--upgrade"])

    print("Upgrading pip...")
    run([str(pip_exe), "install", "--upgrade", "pip"])

    print(f"Installing dependencies from {requirements_file}...")
    run([str(pip_exe), "install", "-r", str(requirements_file)])

    print("\nDone! Activate the environment with:")
    if os.name == "nt":
        print(r"  .\.venv\Scripts\activate")
    else:
        print("  source .venv/bin/activate")


if __name__ == "__main__":
    main()
