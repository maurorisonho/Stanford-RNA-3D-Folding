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
Stanford RNA 3D Folding - Submission Helper

Uploads a submission file to Kaggle and prints the resulting status message.
Useful when preparing late submissions or re-submitting improved solutions.

"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

COMPETITION = "stanford-rna-3d-folding"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a CLI command and capture stdout/stderr."""
    print(f"[run] {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )


def kaggle_cli() -> Path:
    """Locate the Kaggle CLI binary."""
    path = shutil.which("kaggle")
    if not path:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with 'pip install kaggle' inside your "
            "virtual environment."
        )
    return Path(path)


def verify_credentials() -> None:
    """Ensure the user configured the Kaggle credentials file."""
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if creds.exists():
        return
    raise RuntimeError(
        "Missing ~/.kaggle/kaggle.json. Visit https://www.kaggle.com/settings/account "
        "and create a new API token, then place the file in ~/.kaggle/."
    )


def submit(submission_path: Path, message: str) -> None:
    """Submit the file to Kaggle and print the response."""
    kaggle = kaggle_cli()
    verify_credentials()

    cmd = [
        str(kaggle),
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(submission_path),
        "-m",
        message,
    ]

    result = run(cmd)
    print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a prediction file to the Kaggle competition."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the CSV file to submit.",
    )
    parser.add_argument(
        "-m",
        "--message",
        default="Late submission",
        help="Submission message shown on Kaggle (default: 'Late submission').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.file.exists():
        raise FileNotFoundError(f"Submission file not found: {args.file}")

    submit(args.file, args.message)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[error] Kaggle CLI failed with exit code {exc.returncode}")
        if exc.stdout:
            print(exc.stdout.strip())
        if exc.stderr:
            print(exc.stderr.strip())
        sys.exit(exc.returncode)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] {exc}")
        sys.exit(1)
