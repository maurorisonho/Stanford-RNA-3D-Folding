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
Usage Examples - Environment Manager
===================================

This file contains practical examples of how to use environment_manager.py
in different scenarios for the Stanford RNA 3D Folding project.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command_list):
    """Execute command and show result"""
    print(f"$ python environment_manager.py {' '.join(command_list)}")
    print("-" * 60)
    
    script_dir = Path(__file__).parent
    env_manager = script_dir / "environment_manager.py"
    
    cmd = [sys.executable, str(env_manager)] + command_list
    result = subprocess.run(cmd, capture_output=False)
    
    print("-" * 60)
    print(f"Exit code: {result.returncode}")
    print()
    
    return result.returncode == 0

def main():
    print("[RNA] USAGE EXAMPLES - ENVIRONMENT MANAGER")
    print("=" * 60)
    print()
    
    examples = [
        ("Check Environment Status", ["check"]),
        ("Show Project Information", ["info"]),
        ("List Packages (compact format)", ["list"]),
        ("List Packages (detailed format)", ["list", "--detailed"]),
        ("Test Critical Imports", ["test"]),
        ("Generate Requirements Freeze", ["freeze"]),
        ("Generate Custom Freeze", ["freeze", "--output", "backup_requirements.txt"])
    ]
    
    print("Choose an example to run:")
    print()
    for i, (description, _) in enumerate(examples, 1):
        print(f"{i}. {description}")
    
    print("0. Run all examples")
    print()
    
    try:
        choice = input("Enter your choice (0-7): ").strip()
        
        if choice == "0":
            # Run all examples
            for i, (description, command) in enumerate(examples, 1):
                print(f"\n{'='*60}")
                print(f"EXAMPLE {i}: {description}")
                print('='*60)
                run_command(command)
                
                if i < len(examples):
                    input("Press Enter to continue to the next example...")
        
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            description, command = examples[idx]
            
            print(f"\n{'='*60}")
            print(f"EXAMPLE: {description}")
            print('='*60)
            run_command(command)
        
        else:
            print("[ERROR] Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n[WARNING]  Operation cancelled by user.")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

def show_scenarios():
    """Show common usage scenarios"""
    print("[TOOLS] COMMON USAGE SCENARIOS")
    print("=" * 40)
    print()
    
    scenarios = {
        "Initial Setup (First time)": [
            "python environment_manager.py check",
            "python environment_manager.py create", 
            "python environment_manager.py install",
            "python environment_manager.py test"
        ],
        
        "Quick Check": [
            "python environment_manager.py check",
            "python environment_manager.py test"
        ],
        
        "Complete Reinstallation": [
            "python environment_manager.py clean",
            "python environment_manager.py create",
            "python environment_manager.py install",
            "python environment_manager.py test"
        ],
        
        "Dependencies Backup": [
            "python environment_manager.py freeze",
            "python environment_manager.py freeze --output backup_$(date +%Y%m%d).txt"
        ],
        
        "Problem Diagnosis": [
            "python environment_manager.py check",
            "python environment_manager.py list",
            "python environment_manager.py test"
        ],
        
        "Cleanup and Maintenance": [
            "python environment_manager.py clean",
            "# Optional: python environment_manager.py create",
            "# Optional: python environment_manager.py install"
        ]
    }
    
    for scenario, commands in scenarios.items():
        print(f"[INFO] {scenario}:")
        for cmd in commands:
            print(f"   {cmd}")
        print()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run interactive examples")
    print("2. Show usage scenarios")
    print()
    
    choice = input("Enter your choice (1-2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        show_scenarios()
    else:
        print("[ERROR] Invalid choice!")