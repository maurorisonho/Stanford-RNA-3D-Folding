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
Quick Setup Script - Stanford RNA 3D Folding
===========================================

Quick script for initial project setup.
Run this file to automatically configure the entire environment.

Usage: python quick_setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_environment_manager(command):
    """Execute environment_manager.py command"""
    script_dir = Path(__file__).parent
    env_manager = script_dir / "environment_manager.py"
    
    cmd = [sys.executable, str(env_manager), command]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    print("[LAUNCH] QUICK SETUP - STANFORD RNA 3D FOLDING")
    print("=" * 50)
    
    # 1. Check initial status
    print("[INFO] 1. Checking initial status...")
    run_environment_manager("check")
    
    input("\n  Press Enter to continue with environment creation...")
    
    # 2. Create virtual environment
    print("\n🔨 2. Creating virtual environment...")
    if not run_environment_manager("create"):
        print("[ERROR] Failed to create virtual environment!")
        sys.exit(1)
    
    # 3. Install dependencies
    print("\n[PACKAGE] 3. Installing dependencies...")
    if not run_environment_manager("install"):
        print("[ERROR] Failed to install dependencies!")
        sys.exit(1)
    
    # 4. Test imports
    print("\n[TEST] 4. Testing imports...")
    if not run_environment_manager("test"):
        print("[WARNING]  Some imports failed, but environment is functional.")
    else:
        print("[OK] All imports worked!")
    
    # 5. Final status
    print("\n[RESULT] 5. Final environment status:")
    run_environment_manager("check")
    
    print("\n[SUCCESS] SETUP COMPLETED SUCCESSFULLY!")
    print("\nNext steps:")
    print("1. Activate environment: source stanford_rna3d/.venv/bin/activate")
    print("2. Run Jupyter: jupyter lab")
    print("3. Or use: python environment_manager.py [command]")

if __name__ == "__main__":
    main()