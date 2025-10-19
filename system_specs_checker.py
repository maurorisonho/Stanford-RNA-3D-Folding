#!/usr/bin/env python3
"""
System Specifications Checker

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Created: October 19, 2025
License: MIT License

A comprehensive Python script to check laptop hardware and system configurations
for machine learning development environments.
"""

import subprocess
import platform
import psutil
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class SystemSpecsChecker:
    """Comprehensive system specifications checker for laptops."""
    
    def __init__(self):
        """Initialize the system checker."""
        self.specs = {}
        self.errors = []
        
    def run_command(self, command: str) -> Optional[str]:
        """Execute a shell command and return output."""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.errors.append(f"Command failed: {command} - {str(e)}")
            return None
    
    def get_system_info(self) -> Dict:
        """Get basic system information."""
        info = {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Get detailed OS information
        hostnamectl = self.run_command('hostnamectl')
        if hostnamectl:
            for line in hostnamectl.split('\n'):
                if 'Operating System:' in line:
                    info['os_name'] = line.split(':', 1)[1].strip()
                elif 'Kernel:' in line:
                    info['kernel'] = line.split(':', 1)[1].strip()
                elif 'Hardware Vendor:' in line:
                    info['vendor'] = line.split(':', 1)[1].strip()
                elif 'Hardware Model:' in line:
                    info['model'] = line.split(':', 1)[1].strip()
                elif 'Chassis:' in line:
                    info['chassis'] = line.split(':', 1)[1].strip()
        
        return info
    
    def get_cpu_info(self) -> Dict:
        """Get detailed CPU information."""
        cpu_info = {
            'model': platform.processor(),
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'current_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'min_freq': psutil.cpu_freq().min if psutil.cpu_freq() else None,
            'max_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'usage_percent': psutil.cpu_percent(interval=1)
        }
        
        # Get detailed CPU information from lscpu
        lscpu = self.run_command('lscpu')
        if lscpu:
            for line in lscpu.split('\n'):
                if 'Model name:' in line:
                    cpu_info['detailed_model'] = line.split(':', 1)[1].strip()
                elif 'L3 cache:' in line:
                    cpu_info['l3_cache'] = line.split(':', 1)[1].strip()
                elif 'CPU max MHz:' in line:
                    try:
                        cpu_info['max_mhz'] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif 'CPU min MHz:' in line:
                    try:
                        cpu_info['min_mhz'] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
        
        return cpu_info
    
    def get_memory_info(self) -> Dict:
        """Get detailed memory information."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percentage': memory.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'swap_used_gb': round(swap.used / (1024**3), 2),
            'swap_percentage': swap.percent
        }
        
        return memory_info
    
    def get_gpu_info(self) -> Dict:
        """Get GPU information."""
        gpu_info = {'integrated': [], 'dedicated': []}
        
        # Get GPU information from lspci
        lspci = self.run_command('lspci | grep -E "VGA|Display|3D"')
        if lspci:
            for line in lspci.split('\n'):
                if 'Intel' in line:
                    gpu_info['integrated'].append(line.split(': ', 1)[1] if ': ' in line else line)
                elif 'NVIDIA' in line or 'AMD' in line or 'ATI' in line:
                    gpu_info['dedicated'].append(line.split(': ', 1)[1] if ': ' in line else line)
        
        # Get NVIDIA GPU details if available
        nvidia_smi = self.run_command('nvidia-smi --query-gpu=name,memory.total,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits')
        if nvidia_smi:
            nvidia_details = []
            for line in nvidia_smi.split('\n'):
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 5:
                        nvidia_details.append({
                            'name': parts[0],
                            'memory_mb': parts[1],
                            'temperature_c': parts[2],
                            'power_w': parts[3],
                            'utilization_percent': parts[4]
                        })
            gpu_info['nvidia_details'] = nvidia_details
        
        # Get NVIDIA driver version
        nvidia_version = self.run_command('nvidia-smi --query-gpu=driver_version --format=csv,noheader')
        if nvidia_version:
            gpu_info['nvidia_driver'] = nvidia_version.strip()
        
        return gpu_info
    
    def get_storage_info(self) -> Dict:
        """Get storage information."""
        storage_info = {'partitions': []}
        
        # Get disk usage for root
        disk_usage = psutil.disk_usage('/')
        storage_info['root_partition'] = {
            'total_gb': round(disk_usage.total / (1024**3), 2),
            'used_gb': round(disk_usage.used / (1024**3), 2),
            'free_gb': round(disk_usage.free / (1024**3), 2),
            'percentage': round((disk_usage.used / disk_usage.total) * 100, 1)
        }
        
        return storage_info
    
    def get_battery_info(self) -> Dict:
        """Get battery information."""
        battery_info = {}
        
        # Get battery information using psutil
        battery = psutil.sensors_battery()
        if battery:
            battery_info = {
                'percentage': battery.percent,
                'plugged_in': battery.power_plugged,
                'time_left_seconds': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
            }
        
        return battery_info
    
    def get_temperature_info(self) -> Dict:
        """Get temperature information."""
        temp_info = {}
        
        # Get temperature from sensors command
        sensors = self.run_command('sensors')
        if sensors:
            # Parse CPU temperatures
            cpu_temps = []
            for line in sensors.split('\n'):
                if 'Core' in line and '°C' in line:
                    try:
                        temp_match = re.search(r'\+(\d+\.\d+)°C', line)
                        if temp_match:
                            cpu_temps.append(float(temp_match.group(1)))
                    except (ValueError, AttributeError):
                        pass
                elif 'Package id 0:' in line and '°C' in line:
                    try:
                        temp_match = re.search(r'\+(\d+\.\d+)°C', line)
                        if temp_match:
                            temp_info['cpu_package_temp'] = float(temp_match.group(1))
                    except (ValueError, AttributeError):
                        pass
            
            if cpu_temps:
                temp_info['cpu_core_temps'] = cpu_temps
                temp_info['cpu_avg_temp'] = round(sum(cpu_temps) / len(cpu_temps), 1)
        
        return temp_info
    
    def check_ml_environment(self) -> Dict:
        """Check machine learning environment readiness."""
        ml_info = {
            'python_packages': {},
            'cuda_available': False,
            'recommendations': []
        }
        
        # Check important Python packages
        packages_to_check = ['numpy', 'pandas', 'torch', 'tensorflow', 'scikit-learn']
        
        for package in packages_to_check:
            try:
                result = subprocess.run(
                    ['python', '-c', f'import {package}; print({package}.__version__)'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    ml_info['python_packages'][package] = result.stdout.strip()
                else:
                    ml_info['python_packages'][package] = 'Not installed'
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                ml_info['python_packages'][package] = 'Error checking'
        
        # Check CUDA availability
        try:
            result = subprocess.run(
                ['python', '-c', 'import torch; print(torch.cuda.is_available())'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                ml_info['cuda_available'] = result.stdout.strip() == 'True'
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return ml_info
    
    def run_full_check(self) -> Dict:
        """Run complete system check."""
        print("Starting comprehensive system check...")
        
        # Collect all system information
        self.specs['system'] = self.get_system_info()
        print("System information collected")
        
        self.specs['cpu'] = self.get_cpu_info()
        print("CPU information collected")
        
        self.specs['memory'] = self.get_memory_info()
        print("Memory information collected")
        
        self.specs['gpu'] = self.get_gpu_info()
        print("GPU information collected")
        
        self.specs['storage'] = self.get_storage_info()
        print("Storage information collected")
        
        self.specs['battery'] = self.get_battery_info()
        print("Battery information collected")
        
        self.specs['temperature'] = self.get_temperature_info()
        print("Temperature information collected")
        
        self.specs['ml_environment'] = self.check_ml_environment()
        print("ML environment checked")
        
        if self.errors:
            self.specs['errors'] = self.errors
        
        return self.specs
    
    def format_output(self) -> str:
        """Format system specifications for display."""
        if not self.specs:
            return "No system information collected."
        
        output = []
        output.append("LAPTOP SYSTEM SPECIFICATIONS")
        output.append("=" * 50)
        
        # System Information
        if 'system' in self.specs:
            sys_info = self.specs['system']
            output.append("\nSYSTEM INFORMATION")
            output.append("-" * 30)
            output.append(f"OS: {sys_info.get('os_name', 'Unknown')}")
            output.append(f"Kernel: {sys_info.get('kernel', 'Unknown')}")
            output.append(f"Architecture: {sys_info.get('architecture', 'Unknown')}")
            output.append(f"Hostname: {sys_info.get('hostname', 'Unknown')}")
            if 'vendor' in sys_info and 'model' in sys_info:
                output.append(f"Hardware: {sys_info['vendor']} {sys_info['model']}")
        
        # CPU Information
        if 'cpu' in self.specs:
            cpu_info = self.specs['cpu']
            output.append(f"\nPROCESSOR")
            output.append("-" * 30)
            output.append(f"Model: {cpu_info.get('detailed_model', 'Unknown')}")
            output.append(f"Cores: {cpu_info.get('physical_cores', '?')} physical / {cpu_info.get('logical_cores', '?')} logical")
            if cpu_info.get('min_mhz') and cpu_info.get('max_mhz'):
                output.append(f"Frequency: {cpu_info['min_mhz']:.0f}MHz - {cpu_info['max_mhz']:.0f}MHz")
        
        # Memory Information
        if 'memory' in self.specs:
            mem_info = self.specs['memory']
            output.append(f"\nMEMORY")
            output.append("-" * 30)
            output.append(f"Total RAM: {mem_info.get('total_gb', '?')} GB")
            output.append(f"Available: {mem_info.get('available_gb', '?')} GB")
            output.append(f"Used: {mem_info.get('used_gb', '?')} GB ({mem_info.get('percentage', '?')}%)")
        
        # GPU Information
        if 'gpu' in self.specs:
            gpu_info = self.specs['gpu']
            output.append(f"\nGRAPHICS")
            output.append("-" * 30)
            
            if gpu_info.get('integrated'):
                output.append("Integrated GPU:")
                for gpu in gpu_info['integrated']:
                    output.append(f"  - {gpu}")
            
            if gpu_info.get('dedicated'):
                output.append("Dedicated GPU:")
                for gpu in gpu_info['dedicated']:
                    output.append(f"  - {gpu}")
            
            if gpu_info.get('nvidia_details'):
                for detail in gpu_info['nvidia_details']:
                    output.append(f"  Memory: {detail.get('memory_mb', '?')}MB")
                    output.append(f"  Temperature: {detail.get('temperature_c', '?')}°C")
            
            if gpu_info.get('nvidia_driver'):
                output.append(f"NVIDIA Driver: {gpu_info['nvidia_driver']}")
        
        # Storage Information
        if 'storage' in self.specs:
            storage_info = self.specs['storage']
            output.append(f"\nSTORAGE")
            output.append("-" * 30)
            
            root = storage_info.get('root_partition', {})
            output.append(f"Root Partition: {root.get('used_gb', '?')}/{root.get('total_gb', '?')} GB ({root.get('percentage', '?')}% used)")
            output.append(f"Free Space: {root.get('free_gb', '?')} GB")
        
        # Battery Information  
        if 'battery' in self.specs and self.specs['battery']:
            battery_info = self.specs['battery']
            output.append(f"\nBATTERY")
            output.append("-" * 30)
            output.append(f"Charge: {battery_info.get('percentage', '?')}%")
            output.append(f"Plugged In: {battery_info.get('plugged_in', 'Unknown')}")
        
        # Temperature Information
        if 'temperature' in self.specs:
            temp_info = self.specs['temperature']
            output.append(f"\nTEMPERATURES")
            output.append("-" * 30)
            
            if temp_info.get('cpu_package_temp'):
                output.append(f"CPU Package: {temp_info['cpu_package_temp']:.1f}°C")
            
            if temp_info.get('cpu_avg_temp'):
                output.append(f"CPU Average: {temp_info['cpu_avg_temp']:.1f}°C")
        
        # ML Environment
        if 'ml_environment' in self.specs:
            ml_info = self.specs['ml_environment']
            output.append(f"\nMACHINE LEARNING ENVIRONMENT")
            output.append("-" * 30)
            
            # Key packages
            key_packages = ['torch', 'tensorflow', 'numpy', 'pandas', 'scikit-learn']
            for pkg in key_packages:
                if pkg in ml_info['python_packages']:
                    version = ml_info['python_packages'][pkg]
                    status = "[OK]" if version != 'Not installed' else "[NO]"
                    output.append(f"{status} {pkg}: {version}")
            
            if ml_info.get('cuda_available'):
                output.append("[OK] CUDA: Available")
            else:
                output.append("[NO] CUDA: Not available")
        
        output.append(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return '\n'.join(output)
    
    def save_report(self, filename: str = None):
        """Save the report to a file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"system_specs_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.format_output())
        
        print(f"\nReport saved to: {filename}")
        return filename

def main():
    """Main function to run the system checker."""
    print("System Specifications Checker")
    print("=" * 40)
    
    checker = SystemSpecsChecker()
    
    try:
        # Run full system check
        specs = checker.run_full_check()
        
        # Display formatted report
        print("\n" + checker.format_output())
        
        # Ask user if they want to save reports
        save_reports = input("\nSave detailed reports? (y/N): ").lower().strip()
        if save_reports == 'y':
            checker.save_report()
    
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user.")
    except Exception as e:
        print(f"\nError during system check: {e}")

if __name__ == "__main__":
    main()