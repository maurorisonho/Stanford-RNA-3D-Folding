#!/usr/bin/env python3
"""
Example usage of SystemSpecsChecker

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
Created: October 19, 2025
License: MIT License

This file demonstrates different ways to use the SystemSpecsChecker class
for specific system monitoring needs.
"""

from system_specs_checker import SystemSpecsChecker
import json

def quick_ml_check():
    """Quick check for ML environment readiness."""
    print("Quick ML Environment Check")
    print("=" * 35)
    
    checker = SystemSpecsChecker()
    
    # Get specific information needed for ML
    cpu_info = checker.get_cpu_info()
    memory_info = checker.get_memory_info()
    gpu_info = checker.get_gpu_info()
    ml_env = checker.check_ml_environment()
    
    print(f"CPU: {cpu_info.get('detailed_model', 'Unknown')}")
    print(f"Cores: {cpu_info.get('physical_cores', '?')} physical / {cpu_info.get('logical_cores', '?')} logical")
    print(f"RAM: {memory_info.get('total_gb', '?')} GB total, {memory_info.get('available_gb', '?')} GB available")
    
    if gpu_info.get('nvidia_details'):
        nvidia = gpu_info['nvidia_details'][0]
        print(f"GPU: {nvidia.get('name', 'Unknown')} ({nvidia.get('memory_mb', '?')}MB)")
    elif gpu_info.get('dedicated'):
        print(f"GPU: {gpu_info['dedicated'][0]}")
    else:
        print("GPU: Integrated only")
    
    print(f"CUDA: {'Available' if ml_env.get('cuda_available') else 'Not available'}")
    print(f"PyTorch: {ml_env['python_packages'].get('torch', 'Not installed')}")
    
    # ML Readiness Score
    score = 0
    if cpu_info.get('logical_cores', 0) >= 8:
        score += 2
    if memory_info.get('total_gb', 0) >= 16:
        score += 2
    if memory_info.get('total_gb', 0) >= 32:
        score += 1
    if gpu_info.get('nvidia_details'):
        score += 3
    if ml_env.get('cuda_available'):
        score += 2
    
    print(f"\nML Readiness Score: {score}/10")
    if score >= 8:
        print("[OK] Excellent for ML/AI development")
    elif score >= 6:
        print("[OK] Good for most ML tasks")
    elif score >= 4:
        print("[WARN] Suitable for basic ML")
    else:
        print("[ERROR] Limited ML capability")

def monitor_temps():
    """Monitor system temperatures."""
    print("System Temperature Monitor")
    print("=" * 35)
    
    checker = SystemSpecsChecker()
    temp_info = checker.get_temperature_info()
    
    if temp_info.get('cpu_package_temp'):
        temp = temp_info['cpu_package_temp']
        if temp < 70:
            status = "[OK] Normal"
        elif temp < 85:
            status = "[WARN] Warm"
        else:
            status = "[HOT] Hot"
        print(f"CPU Package: {temp:.1f}°C {status}")
    
    if temp_info.get('cpu_avg_temp'):
        print(f"CPU Average: {temp_info['cpu_avg_temp']:.1f}°C")
    
    # GPU temperature from previous check
    gpu_info = checker.get_gpu_info()
    if gpu_info.get('nvidia_details'):
        gpu_temp = float(gpu_info['nvidia_details'][0].get('temperature_c', 0))
        if gpu_temp < 70:
            status = "[OK] Cool"
        elif gpu_temp < 85:
            status = "[WARN] Warm"
        else:
            status = "[HOT] Hot"
        print(f"GPU: {gpu_temp:.0f}°C {status}")

def check_storage_health():
    """Check storage health and usage."""
    print("Storage Health Check")
    print("=" * 35)
    
    checker = SystemSpecsChecker()
    storage_info = checker.get_storage_info()
    
    # Check root partition
    root = storage_info.get('root_partition', {})
    used_pct = root.get('percentage', 0)
    
    if used_pct < 70:
        status = "[OK] Healthy"
    elif used_pct < 85:
        status = "[WARN] Getting full"
    else:
        status = "[ERROR] Almost full"
    
    print(f"Root partition: {used_pct:.1f}% used {status}")
    print(f"Free space: {root.get('free_gb', 0):.1f} GB")

def system_overview():
    """Get a quick system overview."""
    print("System Overview")
    print("=" * 25)
    
    checker = SystemSpecsChecker()
    
    # Collect basic info
    sys_info = checker.get_system_info()
    cpu_info = checker.get_cpu_info()
    mem_info = checker.get_memory_info()
    
    print(f"System: {sys_info.get('os_name', 'Unknown')}")
    print(f"CPU: {cpu_info.get('detailed_model', 'Unknown')}")
    print(f"RAM: {mem_info.get('total_gb', 0):.0f} GB")
    print(f"Usage: CPU {cpu_info.get('usage_percent', 0):.1f}% | RAM {mem_info.get('percentage', 0):.1f}%")

def export_for_sharing():
    """Export system info in a format suitable for sharing (removes sensitive data)."""
    print("Exporting System Info for Sharing")
    print("=" * 40)
    
    checker = SystemSpecsChecker()
    specs = checker.run_full_check()
    
    # Remove sensitive information
    safe_specs = {
        'cpu': {
            'model': specs.get('cpu', {}).get('detailed_model'),
            'cores': f"{specs.get('cpu', {}).get('physical_cores')}c/{specs.get('cpu', {}).get('logical_cores')}t",
            'max_freq': specs.get('cpu', {}).get('max_mhz')
        },
        'memory': {
            'total_gb': specs.get('memory', {}).get('total_gb'),
        },
        'gpu': {
            'integrated': specs.get('gpu', {}).get('integrated'),
            'dedicated': specs.get('gpu', {}).get('dedicated'),
            'nvidia_driver': specs.get('gpu', {}).get('nvidia_driver')
        },
        'os': {
            'name': specs.get('system', {}).get('os_name'),
            'kernel': specs.get('system', {}).get('kernel')
        },
        'ml_ready': specs.get('ml_environment', {}).get('cuda_available', False)
    }
    
    # Save to file
    with open('system_info_shareable.json', 'w') as f:
        json.dump(safe_specs, f, indent=2)
    
    print("[OK] Shareable system info saved to: system_info_shareable.json")
    print("\nSafe to share (no sensitive data):")
    print(json.dumps(safe_specs, indent=2))

if __name__ == "__main__":
    import sys
    
    # Command line interface
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "ml":
            quick_ml_check()
        elif command == "temp":
            monitor_temps()
        elif command == "storage":
            check_storage_health()
        elif command == "overview":
            system_overview()
        elif command == "export":
            export_for_sharing()
        elif command == "full":
            checker = SystemSpecsChecker()
            checker.run_full_check()
            print(checker.format_output())
        else:
            print("Available commands: ml, temp, storage, overview, export, full")
    else:
        # Interactive menu
        print("System Checker - Choose an option:")
        print("1. Quick ML Environment Check")
        print("2. Temperature Monitor")
        print("3. Storage Health Check")
        print("4. System Overview")
        print("5. Export for Sharing")
        print("6. Full System Check")
        print("0. Exit")
        
        try:
            choice = input("\nEnter choice (0-6): ").strip()
            
            if choice == "1":
                quick_ml_check()
            elif choice == "2":
                monitor_temps()
            elif choice == "3":
                check_storage_health()
            elif choice == "4":
                system_overview()
            elif choice == "5":
                export_for_sharing()
            elif choice == "6":
                checker = SystemSpecsChecker()
                checker.run_full_check()
                print(checker.format_output())
            elif choice == "0":
                print("Goodbye!")
            else:
                print("Invalid choice")
        except KeyboardInterrupt:
            print("\nGoodbye!")