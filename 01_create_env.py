#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stanford RNA 3D Folding - Environment Setup Script
Creates Python virtual environment, installs dependencies, and validates setup.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Email: mauro.risonho@gmail.com
Created: 2025-10-18 22:30:00

License: MIT (see LICENSE file at repository root)
Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import List, Optional

# Competition identifier
COMP = "stanford-rna-3d-folding"

# Python version requirements
MIN_PYTHON_VERSION = (3, 9)
RECOMMENDED_PYTHON_VERSION = (3, 13, 5)

# Required system packages
REQUIRED_PACKAGES = [
    "torch>=2.0.0",
    "pytorch-lightning>=2.0.0", 
    "transformers>=4.30.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
    "scikit-learn>=1.3.0",
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
    "ipykernel>=6.23.0",
    "biopython>=1.81",
    "optuna>=3.0.0",
    "wandb>=0.15.0",
    "tqdm>=4.65.0",
    "kaggle>=1.5.13"
]

# Development packages (optional but recommended)
DEV_PACKAGES = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.4.0",
    "pre-commit>=3.3.0"
]


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    current_version = sys.version_info[:3]
    min_version = MIN_PYTHON_VERSION
    recommended_version = RECOMMENDED_PYTHON_VERSION
    
    print(f"[INFO] Python version: {'.'.join(map(str, current_version))}")
    print(f"[INFO] Minimum required: {'.'.join(map(str, min_version))}")
    print(f"[INFO] Recommended: {'.'.join(map(str, recommended_version))}")
    
    if current_version < min_version:
        print(f"[ERROR] Python version {'.'.join(map(str, current_version))} is too old!")
        print(f"[ERROR] Minimum required: {'.'.join(map(str, min_version))}")
        return False
    
    if current_version < recommended_version:
        print(f"[WARNING] Using older Python version. Recommended: {'.'.join(map(str, recommended_version))}")
    else:
        print(f"[SUCCESS] Python version meets all requirements!")
    
    return True


def check_system_requirements():
    """Check system requirements and available tools."""
    print(f"\n[INFO] System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python executable: {sys.executable}")
    
    # Check for essential system tools
    required_tools = ['pip', 'git']
    optional_tools = ['cuda', 'nvidia-smi']
    
    print(f"\n[INFO] Checking system tools:")
    for tool in required_tools:
        if shutil.which(tool):
            print(f"   ✓ {tool}: Available")
        else:
            print(f"   ✗ {tool}: NOT FOUND - REQUIRED")
            return False
    
    for tool in optional_tools:
        if shutil.which(tool):
            print(f"   ✓ {tool}: Available")
        else:
            print(f"   - {tool}: Not available (optional)")
    
    # Check CUDA availability
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(f"   ✓ CUDA/GPU: Available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"   - CUDA/GPU: Not available (will use CPU)")
    
    return True


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> bool:
    """Execute command and handle errors gracefully."""
    try:
        print(f"[RUN] {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=check
        )
        
        if result.stdout:
            print(f"[OUTPUT] {result.stdout.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(f"[ERROR] {e.stderr.strip()}")
        return False


def create_virtual_environment(env_path: Path, python_executable: Optional[str] = None) -> bool:
    """Create a Python virtual environment."""
    print(f"\n[INFO] Creating virtual environment at: {env_path}")
    
    if env_path.exists():
        print(f"[WARNING] Virtual environment already exists at {env_path}")
        return True
    
    try:
        # Use specified Python executable or current one
        if python_executable:
            if not shutil.which(python_executable):
                print(f"[ERROR] Python executable not found: {python_executable}")
                return False
            cmd = [python_executable, '-m', 'venv', str(env_path)]
            return run_command(cmd)
        else:
            # Use venv module directly
            venv.create(env_path, with_pip=True)
            print(f"[SUCCESS] Virtual environment created successfully")
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to create virtual environment: {e}")
        return False


def get_activation_script(env_path: Path) -> tuple[Path, str]:
    """Get the activation script path and command for the virtual environment."""
    system = platform.system().lower()
    
    if system == "windows":
        activate_script = env_path / "Scripts" / "activate.bat"
        activate_cmd = str(activate_script)
    else:
        activate_script = env_path / "bin" / "activate"
        activate_cmd = f"source {activate_script}"
    
    return activate_script, activate_cmd


def get_python_executable(env_path: Path) -> Path:
    """Get the Python executable path in the virtual environment."""
    system = platform.system().lower()
    
    if system == "windows":
        return env_path / "Scripts" / "python.exe"
    else:
        return env_path / "bin" / "python"


def install_packages(env_path: Path, packages: List[str], upgrade_pip: bool = True) -> bool:
    """Install packages in the virtual environment."""
    python_exe = get_python_executable(env_path)
    
    if not python_exe.exists():
        print(f"[ERROR] Python executable not found: {python_exe}")
        return False
    
    # Upgrade pip first
    if upgrade_pip:
        print(f"\n[INFO] Upgrading pip...")
        if not run_command([str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip']):
            print(f"[WARNING] Failed to upgrade pip, continuing...")
    
    # Install packages
    print(f"\n[INFO] Installing {len(packages)} packages...")
    
    # Install all packages in one command for better dependency resolution
    cmd = [str(python_exe), '-m', 'pip', 'install'] + packages
    
    if not run_command(cmd):
        print(f"[ERROR] Package installation failed")
        return False
    
    print(f"[SUCCESS] All packages installed successfully")
    return True


def create_requirements_file(project_dir: Path, packages: List[str]) -> bool:
    """Create requirements.txt file."""
    requirements_file = project_dir / "requirements.txt"
    
    print(f"\n[INFO] Creating requirements.txt at: {requirements_file}")
    
    try:
        # Create requirements content with version specifications
        content = [
            "# Stanford RNA 3D Folding - Python Dependencies",
            f"# Generated on: {platform.node()} - {platform.system()}",
            f"# Python version: {'.'.join(map(str, sys.version_info[:3]))}",
            "",
            "# Core ML and Data Science",
        ]
        
        # Categorize packages
        core_ml = [
            "torch>=2.9.0",
            "torchvision>=0.24.0", 
            "pytorch-lightning>=2.5.0",
            "transformers>=4.57.0",
            "scikit-learn>=1.7.0"
        ]
        
        data_science = [
            "pandas>=2.3.0",
            "numpy>=2.3.0",
            "matplotlib>=3.10.0",
            "seaborn>=0.13.0",
            "plotly>=6.3.0"
        ]
        
        bio_packages = [
            "biopython>=1.85"
        ]
        
        optimization = [
            "optuna>=4.5.0",
            "wandb>=0.22.0"
        ]
        
        jupyter_packages = [
            "jupyter>=1.1.0",
            "jupyterlab>=4.4.0",
            "ipykernel>=7.0.0"
        ]
        
        utilities = [
            "tqdm>=4.66.0",
            "kaggle>=1.6.0"
        ]
        
        # Add categorized packages
        categories = [
            ("# Core ML and Deep Learning", core_ml),
            ("# Data Science and Visualization", data_science),
            ("# Bioinformatics", bio_packages),
            ("# Optimization and Tracking", optimization),
            ("# Jupyter and Notebooks", jupyter_packages),
            ("# Utilities", utilities)
        ]
        
        for category_name, category_packages in categories:
            content.append(category_name)
            content.extend(category_packages)
            content.append("")
        
        # Write requirements file
        requirements_file.write_text('\n'.join(content), encoding='utf-8')
        
        print(f"[SUCCESS] Requirements file created with {len(packages)} dependencies")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create requirements.txt: {e}")
        return False


def validate_installation(env_path: Path) -> bool:
    """Validate that key packages are properly installed."""
    print(f"\n[INFO] Validating installation...")
    
    python_exe = get_python_executable(env_path)
    
    # Test key imports
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import pandas as pd; print(f'Pandas: {pd.__version__}')",
        "import numpy as np; print(f'NumPy: {np.__version__}')",
        "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')",
        "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')",
        "import jupyter; print('Jupyter: Available')",
    ]
    
    all_passed = True
    
    for test_import in test_imports:
        cmd = [str(python_exe), '-c', test_import]
        if not run_command(cmd, check=False):
            all_passed = False
    
    # Test CUDA if available
    cuda_test = "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    cmd = [str(python_exe), '-c', cuda_test]
    run_command(cmd, check=False)
    
    if all_passed:
        print(f"[SUCCESS] Installation validation passed!")
    else:
        print(f"[WARNING] Some packages may not be properly installed")
    
    return all_passed


def create_project_structure(project_dir: Path) -> bool:
    """Create basic project directory structure."""
    print(f"\n[INFO] Creating project structure at: {project_dir}")
    
    # Create directories
    dirs_to_create = [
        "data/raw",
        "data/processed", 
        "data/external",
        "notebooks",
        "src",
        "configs",
        "checkpoints",
        "submissions",
        "tests",
        "portfolio"
    ]
    
    try:
        for dir_path in dirs_to_create:
            full_path = project_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files for empty directories
            if not any(full_path.iterdir()):
                (full_path / ".gitkeep").touch()
        
        print(f"[SUCCESS] Project structure created")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create project structure: {e}")
        return False


def create_activation_helper(project_dir: Path, env_path: Path) -> bool:
    """Create helper scripts for environment activation."""
    print(f"\n[INFO] Creating activation helper scripts...")
    
    activate_script, activate_cmd = get_activation_script(env_path)
    
    # Create shell script for Unix systems
    if platform.system().lower() != "windows":
        shell_script = project_dir / "activate_env.sh"
        content = f"""#!/bin/bash
# Stanford RNA 3D Folding - Environment Activation Script

echo "Activating Stanford RNA 3D Folding environment..."
echo "Environment path: {env_path.absolute()}"

# Activate virtual environment
{activate_cmd}

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Environment activated successfully!"
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "✗ Failed to activate environment"
    exit 1
fi

echo ""
echo "Quick start commands:"
echo "  pip list                 # List installed packages"
echo "  jupyter lab             # Start Jupyter Lab"
echo "  python 02_setup_project.py --dest ./stanford_rna3d --lang en"
echo ""
"""
        shell_script.write_text(content, encoding='utf-8')
        shell_script.chmod(0o755)  # Make executable
        
    # Create batch script for Windows
    if platform.system().lower() == "windows":
        batch_script = project_dir / "activate_env.bat"
        content = f"""@echo off
REM Stanford RNA 3D Folding - Environment Activation Script

echo Activating Stanford RNA 3D Folding environment...
echo Environment path: {env_path.absolute()}

REM Activate virtual environment  
call {activate_script}

REM Verify activation
if "%VIRTUAL_ENV%" == "" (
    echo X Failed to activate environment
    pause
    exit /b 1
) else (
    echo √ Environment activated successfully!
    echo Python: %VIRTUAL_ENV%\\Scripts\\python.exe
    python --version
)

echo.
echo Quick start commands:
echo   pip list                 # List installed packages
echo   jupyter lab             # Start Jupyter Lab  
echo   python 02_setup_project.py --dest ./stanford_rna3d --lang en
echo.
"""
        batch_script.write_text(content, encoding='utf-8')
    
    print(f"[SUCCESS] Activation helper scripts created")
    return True


def main():
    """Main function for environment setup."""
    parser = argparse.ArgumentParser(
        description="Setup Python environment for Stanford RNA 3D Folding competition"
    )
    parser.add_argument("--env-name", default=".venv",
                       help="Virtual environment name/path (default: .venv)")
    parser.add_argument("--python", 
                       help="Specific Python executable to use")
    parser.add_argument("--project-dir", default=".",
                       help="Project directory (default: current directory)")
    parser.add_argument("--dev-packages", action="store_true",
                       help="Install development packages (pytest, black, etc.)")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip installation validation")
    parser.add_argument("--force", action="store_true", 
                       help="Force recreate environment if exists")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stanford RNA 3D Folding - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print(f"[ERROR] Python version check failed")
        sys.exit(1)
    
    # Check system requirements
    if not check_system_requirements():
        print(f"[ERROR] System requirements check failed")
        sys.exit(1)
    
    # Setup paths
    project_dir = Path(args.project_dir).resolve()
    env_path = project_dir / args.env_name
    
    print(f"\n[INFO] Project directory: {project_dir}")
    print(f"[INFO] Environment path: {env_path}")
    
    # Create project directory
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle existing environment
    if env_path.exists():
        if args.force:
            print(f"[INFO] Removing existing environment (--force flag)")
            shutil.rmtree(env_path)
        else:
            print(f"[INFO] Virtual environment already exists at {env_path}")
            print(f"[INFO] Use --force to recreate")
    
    # Create virtual environment
    if not env_path.exists():
        if not create_virtual_environment(env_path, args.python):
            print(f"[ERROR] Failed to create virtual environment")
            sys.exit(1)
    
    # Prepare package list
    packages_to_install = REQUIRED_PACKAGES.copy()
    if args.dev_packages:
        packages_to_install.extend(DEV_PACKAGES)
        print(f"[INFO] Including development packages")
    
    # Install packages
    if not install_packages(env_path, packages_to_install):
        print(f"[ERROR] Package installation failed")
        sys.exit(1)
    
    # Create project structure
    if not create_project_structure(project_dir):
        print(f"[WARNING] Failed to create project structure")
    
    # Create requirements.txt
    if not create_requirements_file(project_dir, packages_to_install):
        print(f"[WARNING] Failed to create requirements.txt")
    
    # Create activation helpers
    if not create_activation_helper(project_dir, env_path):
        print(f"[WARNING] Failed to create activation helpers")
    
    # Validate installation
    if not args.skip_validation:
        validate_installation(env_path)
    
    # Success message
    print("\n" + "=" * 60)
    print("[SUCCESS] Environment setup completed!")
    print("=" * 60)
    
    activate_script, activate_cmd = get_activation_script(env_path)
    
    print(f"\nNext steps:")
    print(f"1. Activate environment:")
    print(f"   {activate_cmd}")
    print(f"   # Or run: ./activate_env.sh (Unix) / activate_env.bat (Windows)")
    print(f"")
    print(f"2. Setup project structure:")
    print(f"   python 02_setup_project.py --dest ./stanford_rna3d --lang en")
    print(f"")
    print(f"3. Start development:")
    print(f"   cd stanford_rna3d")
    print(f"   jupyter lab")
    print(f"")
    print(f"Environment details:")
    print(f"   Location: {env_path.absolute()}")
    print(f"   Python: {get_python_executable(env_path)}")
    print(f"   Packages: {len(packages_to_install)} installed")
    
    if args.dev_packages:
        print(f"   Development tools: Included")
    
    print(f"\nCompetition: https://www.kaggle.com/competitions/{COMP}")


if __name__ == "__main__":
    main()