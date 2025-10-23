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
Stanford RNA 3D Folding - Project Bootstrap Script

Creates the canonical directory structure, validates Kaggle credentials,
downloads competition data when possible, and generates lightweight notebook
templates for the exploration workflow.

"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

COMPETITION = "stanford-rna-3d-folding"
PROJECT_ROOT = Path("stanford_rna3d")
DATA_SUBDIRS = ("raw", "processed", "external", "interim")
NOTEBOOKS = {
    "00_competition_overview.ipynb": "# Competition overview\n",
    "01_eda.ipynb": "# Exploratory data analysis\n",
    "02_baseline.ipynb": "# Baseline modelling\n",
    "03_advanced.ipynb": "# Advanced experiments\n",
    "04_submission.ipynb": "# Submission packaging\n",
}


def run(cmd: List[str], cwd: Path | None = None) -> None:
    """Run a subprocess while streaming output."""
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def kaggle_cli() -> Path | None:
    """Return the Kaggle CLI path if installed."""
    path = shutil.which("kaggle")
    if path:
        return Path(path)
    print("[warn] Kaggle CLI not found in PATH; data download will be skipped.")
    return None


def ensure_kaggle_credentials() -> bool:
    """Verify the user has configured Kaggle API credentials."""
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if creds.exists():
        return True
    print(
        "[warn] Missing ~/.kaggle/kaggle.json. "
        "Follow https://www.kaggle.com/settings/account to generate an API token."
    )
    return False


def create_directories() -> None:
    """Create the project directory structure."""
    print("[info] Creating base folders")
    (PROJECT_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "src").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "tests").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "notebooks").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "submissions").mkdir(parents=True, exist_ok=True)

    data_root = PROJECT_ROOT / "data"
    for sub in DATA_SUBDIRS:
        (data_root / sub).mkdir(parents=True, exist_ok=True)
        (data_root / sub / ".gitkeep").touch(exist_ok=True)

    (PROJECT_ROOT / "tests" / ".gitkeep").touch(exist_ok=True)
    (PROJECT_ROOT / "checkpoints" / ".gitkeep").touch(exist_ok=True)


def create_notebook(path: Path, title: str) -> None:
    """Generate a minimal notebook with the desired title."""
    if path.exists():
        return

    template = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [title],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": ".venv (Python 3.13.5)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.5",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path.write_text(json.dumps(template, indent=1))
    print(f"[ok] Created notebook {path}")


def ensure_notebooks() -> None:
    """Create missing notebooks using the NOTEBOOKS mapping."""
    for name, title in NOTEBOOKS.items():
        create_notebook(PROJECT_ROOT / "notebooks" / name, title)


def download_competition(data_dir: Path) -> None:
    """Try to download the competition data into the raw directory."""
    kaggle = kaggle_cli()
    if kaggle is None or not ensure_kaggle_credentials():
        return

    cmd = [
        str(kaggle),
        "competitions",
        "download",
        "--force",
        "--path",
        str(data_dir),
        "-c",
        COMPETITION,
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] Kaggle download failed: {exc}")
        return

    for archive in data_dir.glob("*.zip"):
        print(f"[info] Extracting {archive.name}")
        shutil.unpack_archive(str(archive), str(data_dir))
        archive.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialise directories and optionally pull competition data."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not attempt to download data from Kaggle.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing notebook stubs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_directories()

    if args.force:
        for name, title in NOTEBOOKS.items():
            create_notebook(PROJECT_ROOT / "notebooks" / name, title)
    else:
        ensure_notebooks()

    if not args.skip_download:
        download_competition(PROJECT_ROOT / "data" / "raw")

    print("[done] Project structure is ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] {exc}")
        sys.exit(1)
