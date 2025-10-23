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
Stanford RNA 3D Folding - Environment Manager
============================================

Script to manage the virtual environment and project dependencies.
Allows checking, creating, recreating and listing all necessary libraries.

Usage:
    python environment_manager.py [command] [options]

Commands:
    check      - Check current environment status
    create     - Create new virtual environment
    recreate   - Recreate virtual environment (remove and create again)
    install    - Install dependencies from requirements.txt
    list       - List installed packages
    test       - Test critical imports
    freeze     - Generate list of installed packages (requirements freeze)
    clean      - Clean virtual environment
    info       - Show project information

Author: Auto-generated for Stanford RNA 3D Folding project
Date: October 22, 2025
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import time

class EnvironmentManager:
    """Class to manage virtual environment and dependencies."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the environment manager.
        
        Args:
            project_root: Project root path (default: current directory)
        """
        if project_root:
            self.project_root = Path(project_root).resolve()
        else:
            self.project_root = Path(__file__).parent.resolve()
        
        self.stanford_dir = self.project_root / "stanford_rna3d"
        self.venv_dir = self.stanford_dir / ".venv"
        self.requirements_file = self.stanford_dir / "requirements.txt"
        
        # Define executables based on OS
        if platform.system() == "Windows":
            self.python_exe = self.venv_dir / "Scripts" / "python.exe"
            self.pip_exe = self.venv_dir / "Scripts" / "pip.exe"
            self.activate_script = self.venv_dir / "Scripts" / "activate.bat"
        else:
            self.python_exe = self.venv_dir / "bin" / "python"
            self.pip_exe = self.venv_dir / "bin" / "pip"
            self.activate_script = self.venv_dir / "bin" / "activate"
        
        # Critical libraries for testing
        self.critical_imports = [
            "pandas", "numpy", "matplotlib", "seaborn", "plotly",
            "sklearn", "scipy", "torch", "torchvision", "transformers",
            "pytorch_lightning", "torchmetrics", "optuna", "wandb",
            "Bio", "jupyter", "tqdm", "yaml", "requests"
        ]
    
    def _run_command(self, cmd: List[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Execute system command with error handling.
        
        Args:
            cmd: List with command and arguments
            check: If True, raises exception on error
            capture_output: If True, captures stdout/stderr
        
        Returns:
            Result of executed command
        """
        try:
            print(f"[TOOLS] Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=capture_output,
                text=True,
                cwd=str(self.project_root)
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error executing command: {e}")
            print(f"[ERROR] Error output: {e.stderr}")
            raise
        except FileNotFoundError as e:
            print(f"[ERROR] Command not found: {e}")
            raise
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version.
        
        Returns:
            Tuple (version_ok, version_string)
        """
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        # Required minimum version (Python 3.8+)
        min_version = (3, 8)
        is_compatible = version[:2] >= min_version
        
        return is_compatible, version_str
    
    def check_environment_exists(self) -> bool:
        """Check if virtual environment exists.
        
        Returns:
            True if environment exists and is functional
        """
        return (
            self.venv_dir.exists() and
            self.python_exe.exists() and
            self.pip_exe.exists()
        )
    
    def check_requirements_file(self) -> bool:
        """Check if requirements.txt file exists.
        
        Returns:
            True if file exists
        """
        return self.requirements_file.exists()
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed packages in virtual environment.
        
        Returns:
            Dictionary {package_name: version}
        """
        if not self.check_environment_exists():
            return {}
        
        try:
            result = self._run_command([str(self.pip_exe), "list", "--format=json"])
            packages_data = json.loads(result.stdout)
            return {pkg["name"].lower(): pkg["version"] for pkg in packages_data}
        except Exception as e:
            print(f"[WARNING]  Error getting installed packages: {e}")
            return {}
    
    def create_environment(self, force: bool = False) -> bool:
        """Create virtual environment.
        
        Args:
            force: If True, removes existing environment before creating
        
        Returns:
            True if created successfully
        """
        if self.check_environment_exists() and not force:
            print("[OK] Virtual environment already exists. Use --recreate to recreate.")
            return True
        
        if force and self.venv_dir.exists():
            print("🗑️  Removing existing virtual environment...")
            import shutil
            shutil.rmtree(self.venv_dir)
        
        try:
            print(f"🔨 Creating virtual environment at: {self.venv_dir}")
            self._run_command([sys.executable, "-m", "venv", str(self.venv_dir)])
            
            # Update pip
            print("[PACKAGE] Updating pip...")
            self._run_command([str(self.python_exe), "-m", "pip", "install", "--upgrade", "pip"])
            
            print("[OK] Virtual environment created successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error creating virtual environment: {e}")
            return False
    
    def install_requirements(self) -> bool:
        """Install dependencies from requirements.txt.
        
        Returns:
            True if installed successfully
        """
        if not self.check_environment_exists():
            print("[ERROR] Virtual environment does not exist. Run 'create' first.")
            return False
        
        if not self.check_requirements_file():
            print(f"[ERROR] Requirements.txt file not found at: {self.requirements_file}")
            return False
        
        try:
            print("[PACKAGE] Installing dependencies from requirements.txt...")
            self._run_command([
                str(self.pip_exe), "install", "-r", str(self.requirements_file)
            ], capture_output=False)
            
            print("[OK] Dependencies installed successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error installing dependencies: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check for broken dependencies.
        
        Returns:
            True if all dependencies are OK
        """
        if not self.check_environment_exists():
            print("[ERROR] Virtual environment does not exist.")
            return False
        
        try:
            print("[CHECKING] Checking dependencies...")
            result = self._run_command([str(self.pip_exe), "check"])
            print("[OK] All dependencies are OK!")
            return True
            
        except subprocess.CalledProcessError:
            print("[ERROR] Found broken or conflicting dependencies!")
            return False
    
    def test_critical_imports(self) -> Dict[str, bool]:
        """Test critical library imports.
        
        Returns:
            Dictionary {library: import_success}
        """
        if not self.check_environment_exists():
            print("[ERROR] Virtual environment does not exist.")
            return {}
        
        results = {}
        print("[TEST] Testing critical imports...")
        
        for lib in self.critical_imports:
            try:
                # Special name mapping
                import_map = {
                    "sklearn": "from sklearn import datasets",
                    "Bio": "from Bio import SeqIO",
                    "yaml": "import yaml",
                    "pytorch_lightning": "import pytorch_lightning as pl"
                }
                
                import_cmd = import_map.get(lib, f"import {lib}")
                
                result = self._run_command([
                    str(self.python_exe), "-c", import_cmd
                ])
                results[lib] = True
                print(f"  [OK] {lib}")
                
            except subprocess.CalledProcessError:
                results[lib] = False
                print(f"  [ERROR] {lib}")
        
        success_count = sum(results.values())
        total_count = len(results)
        print(f"\n[RESULT] Result: {success_count}/{total_count} libraries OK")
        
        return results
    
    def freeze_requirements(self, output_file: Optional[str] = None) -> bool:
        """Generate freeze file of installed dependencies.
        
        Args:
            output_file: Output file (default: requirements_freeze.txt)
        
        Returns:
            True if generated successfully
        """
        if not self.check_environment_exists():
            print("[ERROR] Virtual environment does not exist.")
            return False
        
        if not output_file:
            output_file = "requirements_freeze.txt"
        
        output_path = self.project_root / output_file
        
        try:
            print(f"[LOGS] Generating freeze at: {output_path}")
            result = self._run_command([str(self.pip_exe), "freeze"])
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
            
            print("[OK] Freeze file generated successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error generating freeze: {e}")
            return False
    
    def show_status(self) -> None:
        """Show complete environment status."""
        print("=" * 80)
        print("[RNA] STANFORD RNA 3D FOLDING - ENVIRONMENT STATUS")
        print("=" * 80)
        
        # System information
        is_compatible, py_version = self.check_python_version()
        print(f"[PYTHON] Python: {py_version} {'[OK]' if is_compatible else '[ERROR] (minimum: 3.8)'}")
        print(f"[SYSTEM] System: {platform.system()} {platform.machine()}")
        print(f"[PROJECT] Project: {self.project_root}")
        
        # Virtual environment status
        env_exists = self.check_environment_exists()
        print(f"[VENV] Virtual Environment: {'[OK] Exists' if env_exists else '[ERROR] Does not exist'}")
        
        if env_exists:
            print(f"   [LOCATION] Location: {self.venv_dir}")
            print(f"   [PYTHON] Python: {self.python_exe}")
            print(f"   [PACKAGE] Pip: {self.pip_exe}")
        
        # Requirements file
        req_exists = self.check_requirements_file()
        print(f"[INFO] Requirements.txt: {'[OK] Exists' if req_exists else '[ERROR] Not found'}")
        
        if req_exists:
            print(f"   [LOCATION] Location: {self.requirements_file}")
        
        # Installed packages
        if env_exists:
            packages = self.get_installed_packages()
            print(f"[PACKAGE] Installed Packages: {len(packages)} packages")
            
            # Check dependencies
            deps_ok = self.check_dependencies()
            print(f"[CHECKING] Dependencies: {'[OK] OK' if deps_ok else '[ERROR] Problems detected'}")
        
        print("=" * 80)
    
    def list_packages(self, detailed: bool = False) -> None:
        """List installed packages.
        
        Args:
            detailed: If True, shows detailed information
        """
        if not self.check_environment_exists():
            print("[ERROR] Virtual environment does not exist.")
            return
        
        packages = self.get_installed_packages()
        
        if not packages:
            print("[PACKAGE] No packages found in virtual environment.")
            return
        
        print(f"[PACKAGE] INSTALLED PACKAGES ({len(packages)} total)")
        print("-" * 60)
        
        if detailed:
            # Show in tabular format
            for name, version in sorted(packages.items()):
                print(f"{name:<30} {version}")
        else:
            # Show in compact format
            sorted_names = sorted(packages.keys())
            for i in range(0, len(sorted_names), 4):
                row = sorted_names[i:i+4]
                print("  ".join(f"{name:<18}" for name in row))
        
        print("-" * 60)
    
    def show_project_info(self) -> None:
        """Show project information."""
        print("=" * 80)
        print("[RNA] STANFORD RNA 3D FOLDING - PROJECT INFORMATION")
        print("=" * 80)
        print("[LOGS] Description: Kaggle competition project for RNA 3D structure prediction")
        print("[TARGET] Objective: Develop ML models for RNA folding")
        print("[PYTHON] Python: 3.8+ (recommended: 3.13.5)")
        print("[PACKAGE] Main Libraries:")
        print("   • Data Analysis: pandas, numpy, matplotlib, seaborn, plotly")
        print("   • Machine Learning: scikit-learn, scipy")
        print("   • Deep Learning: torch, torchvision, transformers, pytorch-lightning")
        print("   • Optimization: optuna, wandb")
        print("   • Bioinformatics: biopython")
        print("   • Development: jupyter, ipykernel, tqdm")
        print("")
        print("[LAUNCH] Quick Commands:")
        print("   python environment_manager.py check      # Check environment")
        print("   python environment_manager.py create     # Create environment")
        print("   python environment_manager.py install    # Install dependencies")
        print("   python environment_manager.py test       # Test imports")
        print("=" * 80)

def main():
    """Main script function."""
    parser = argparse.ArgumentParser(
        description="Environment Manager - Stanford RNA 3D Folding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python environment_manager.py check                    # Check status
  python environment_manager.py create                   # Create environment
  python environment_manager.py recreate                 # Recreate environment
  python environment_manager.py install                  # Install dependencies
  python environment_manager.py list                     # List packages
  python environment_manager.py list --detailed          # List with details
  python environment_manager.py test                     # Test imports
  python environment_manager.py freeze                   # Generate requirements freeze
  python environment_manager.py clean                    # Clean environment
  python environment_manager.py info                     # Project info
        """
    )
    
    parser.add_argument(
        "command",
        choices=["check", "create", "recreate", "install", "list", "test", "freeze", "clean", "info"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information (for 'list' command)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file (for 'freeze' command)"
    )
    
    parser.add_argument(
        "--project-root",
        help="Project root path (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    try:
        manager = EnvironmentManager(args.project_root)
    except Exception as e:
        print(f"[ERROR] Error initializing manager: {e}")
        sys.exit(1)
    
    # Execute command
    start_time = time.time()
    
    try:
        if args.command == "check":
            manager.show_status()
        
        elif args.command == "create":
            success = manager.create_environment()
            sys.exit(0 if success else 1)
        
        elif args.command == "recreate":
            success = manager.create_environment(force=True)
            sys.exit(0 if success else 1)
        
        elif args.command == "install":
            success = manager.install_requirements()
            sys.exit(0 if success else 1)
        
        elif args.command == "list":
            manager.list_packages(detailed=args.detailed)
        
        elif args.command == "test":
            results = manager.test_critical_imports()
            failed_imports = [lib for lib, success in results.items() if not success]
            sys.exit(0 if not failed_imports else 1)
        
        elif args.command == "freeze":
            success = manager.freeze_requirements(args.output)
            sys.exit(0 if success else 1)
        
        elif args.command == "clean":
            if manager.venv_dir.exists():
                import shutil
                print(f"🗑️  Removing virtual environment: {manager.venv_dir}")
                shutil.rmtree(manager.venv_dir)
                print("[OK] Virtual environment removed successfully!")
            else:
                print("ℹ️  No virtual environment found to remove.")
        
        elif args.command == "info":
            manager.show_project_info()
    
    except KeyboardInterrupt:
        print("\n[WARNING]  Operation cancelled by user.")
        sys.exit(130)
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
    
    # Show execution time
    elapsed_time = time.time() - start_time
    print(f"\n[TIME]  Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()