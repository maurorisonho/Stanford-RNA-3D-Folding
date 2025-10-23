# Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
# Created: October 18, 2025 at 14:30:00
# License: MIT License
# Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding
#
# ---
#
# MIT License
#
# Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---

#!/usr/bin/env python3

"""
Stanford RNA 3D Folding - Environment Bootstrapper

Utility script that validates the local Python version, creates a virtual
environment, and installs the project requirements.

"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

MIN_PY_VERSION = (3, 13)
DEFAULT_VENV = Path(".venv")
REQUIREMENTS = Path("stanford_rna3d/requirements.txt")


def run(cmd: List[str], *, env: dict | None = None) -> None:
    """Run a command and stream output; raise on failure."""
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def ensure_python_version() -> None:
    """Guard against running with an unsupported interpreter."""
    if sys.version_info < MIN_PY_VERSION:
        version = ".".join(str(part) for part in sys.version_info[:3])
        required = ".".join(str(part) for part in (*MIN_PY_VERSION, 0))
        raise RuntimeError(
            f"Python {required}+ is required, but interpreter {version} is active."
        )


def create_virtualenv(venv_path: Path, python_bin: str | None) -> Path:
    """Create the virtual environment when it does not exist."""
    if venv_path.exists():
        print(f"[skip] Virtual environment already exists at {venv_path}")
        return venv_path

    python = python_bin or sys.executable
    run([python, "-m", "venv", str(venv_path)])
    print(f"[ok] Created virtual environment at {venv_path}")
    return venv_path


def ensure_activation_hint(venv_path: Path) -> None:
    """Print a helpful activation hint for shells the user might be running."""
    activate = venv_path / "bin" / "activate"
    if not activate.exists():
        print("[warn] Could not find the activation script; check the venv manually.")
        return

    print("\nTo activate the environment run:\n")
    print(f"    source {activate}")
    print("\nPowerShell users:\n")
    print(f"    {venv_path / 'Scripts' / 'Activate.ps1'}")
    print()


def install_requirements(venv_path: Path, extras: Iterable[str]) -> None:
    """Install pinned project requirements plus optional extras."""
    pip = venv_path / "bin" / "pip"
    if not pip.exists():
        raise RuntimeError(f"pip not found at {pip}; virtual environment broken?")

    # Always upgrade pip tooling first for newer manylinux wheels.
    run([str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"])

    if REQUIREMENTS.exists():
        run([str(pip), "install", "-r", str(REQUIREMENTS)])
    else:
        print("[warn] Requirements file missing; skipping dependency installation.")

    extras = list(extras)
    if extras:
        run([str(pip), "install", *extras])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Python 3.13 virtual environment for the project."
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV),
        help="Path to the virtual environment directory (default: .venv).",
    )
    parser.add_argument(
        "--python",
        help="Path to the python executable to use when creating the venv.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Additional packages to install after the base requirements.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Remove the existing virtual environment before recreating it.",
    )
    return parser.parse_args()


def maybe_remove_existing(venv_path: Path, recreate: bool) -> None:
    if recreate and venv_path.exists():
        print(f"[info] Removing existing virtual environment at {venv_path}")
        shutil.rmtree(venv_path)


def main() -> None:
    ensure_python_version()
    args = parse_args()

    venv_path = Path(args.venv).resolve()
    maybe_remove_existing(venv_path, args.recreate)

    create_virtualenv(venv_path, args.python)
    install_requirements(venv_path, args.extra)
    ensure_activation_hint(venv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] {exc}")
        sys.exit(1)
