#!/usr/bin/env python3
"""
Simple GPU Detection Test
Tests basic GPU detection without complex dependencies
"""

import logging
import sys
import os
import subprocess
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_gpus_simple():
    """Simple GPU detection using system commands"""
    print("🔍 Simple GPU Detection Test")
    print("=" * 50)
    
    gpus_found = []
    
    # Method 1: Windows WMIC (Windows only)
    if sys.platform == 'win32':
        print("\n🖥️ Checking Windows GPU Information...")
        gpus_found.extend(detect_gpus_wmic())
    
    # Method 2: Check for NVIDIA drivers
    print("\n🟢 Checking NVIDIA GPU Support...")
    nvidia_info = check_nvidia_support()
    if nvidia_info:
        gpus_found.extend(nvidia_info)
    
    # Method 3: Check for AMD drivers  
    print("\n🔴 Checking AMD GPU Support...")
    amd_info = check_amd_support()
    if amd_info:
        gpus_found.extend(amd_info)
    
    # Summary
    print(f"\n📊 Detection Summary")
    print("=" * 50)
    
    if gpus_found:
        print(f"   ✅ Found {len(gpus_found)} GPU(s)")
        
        nvidia_count = sum(1 for gpu in gpus_found if gpu.get('vendor') == 'NVIDIA')
        amd_count = sum(1 for gpu in gpus_found if gpu.get('vendor') == 'AMD')
        
        print(f"   🟢 NVIDIA GPUs: {nvidia_count}")
        print(f"   🔴 AMD GPUs: {amd_count}")
        
        for i, gpu in enumerate(gpus_found):
            print(f"\n   GPU {i+1}: {gpu.get('name', 'Unknown')}")
            print(f"      Vendor: {gpu.get('vendor', 'Unknown')}")
            if gpu.get('memory_mb'):
                print(f"      Memory: {gpu['memory_mb']:,} MB ({gpu['memory_mb']/1024:.1f} GB)")
            if gpu.get('driver_version'):
                print(f"      Driver: {gpu['driver_version']}")
            
            # AI recommendations
            memory_gb = gpu.get('memory_mb', 0) / 1024
            print(f"      AI Capability: {assess_ai_capability(gpu['vendor'], memory_gb)}")
    else:
        print("   ⚪ No GPUs detected or drivers not installed")
    
    # AI framework recommendations
    if gpus_found:
        print(f"\n🤖 AI Framework Recommendations")
        print("=" * 50)
        
        has_nvidia = any(gpu.get('vendor') == 'NVIDIA' for gpu in gpus_found)
        has_amd = any(gpu.get('vendor') == 'AMD' for gpu in gpus_found)
        
        if has_nvidia:
            print("   🟢 NVIDIA GPU(s) detected:")
            print("      • Install PyTorch with CUDA: pip install torch torchvision torchaudio")
            print("      • Install TensorFlow-GPU: pip install tensorflow[and-cuda]")
            print("      • Use CUDA Toolkit for development")
        
        if has_amd:
            print("   🔴 AMD GPU(s) detected:")
            print("      • Install PyTorch-DirectML: pip install pytorch-directml")
            print("      • Install ONNX Runtime with DirectML: pip install onnxruntime-directml")
            print("      • Update to AMD Adrenalin Edition 23.40.27.06+ for AI optimizations")
    
    # Save results
    results = {
        'detected_gpus': gpus_found,
        'detection_time': datetime.now().isoformat(),
        'system': sys.platform
    }
    
    with open('simple_gpu_detection.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: simple_gpu_detection.json")

def detect_gpus_wmic():
    """Detect GPUs using Windows WMIC"""
    gpus = []
    
    try:
        # Run WMIC command to get GPU information
        cmd = 'wmic path win32_VideoController get name,AdapterRAM,DriverVersion /format:csv'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Skip header line and empty lines
            for line in lines[1:]:
                if line.strip() and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        name = parts[3].strip() if len(parts) > 3 else ''
                        memory_str = parts[1].strip() if len(parts) > 1 else '0'
                        driver = parts[2].strip() if len(parts) > 2 else ''
                        
                        if name and ('NVIDIA' in name or 'AMD' in name or 'Radeon' in name or 'GeForce' in name or 'RTX' in name):
                            # Convert memory to MB
                            memory_mb = 0
                            try:
                                if memory_str and memory_str != '':
                                    memory_mb = int(memory_str) // (1024 * 1024)
                            except:
                                pass
                            
                            # Determine vendor
                            vendor = 'Unknown'
                            if 'NVIDIA' in name or 'GeForce' in name or 'RTX' in name:
                                vendor = 'NVIDIA'
                            elif 'AMD' in name or 'Radeon' in name:
                                vendor = 'AMD'
                            
                            gpu_info = {
                                'name': name,
                                'vendor': vendor,
                                'memory_mb': memory_mb,
                                'driver_version': driver,
                                'detection_method': 'wmic'
                            }
                            gpus.append(gpu_info)
            
            print(f"   ✅ WMIC found {len(gpus)} GPU(s)")
        else:
            print("   ⚠️ WMIC command failed")
    
    except Exception as e:
        print(f"   ❌ WMIC detection failed: {e}")
    
    return gpus

def check_nvidia_support():
    """Check for NVIDIA GPU support"""
    gpus = []
    
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 3:
                        name = parts[0]
                        memory_mb = int(parts[1]) if parts[1].isdigit() else 0
                        driver = parts[2]
                        
                        gpu_info = {
                            'name': name,
                            'vendor': 'NVIDIA',
                            'memory_mb': memory_mb,
                            'driver_version': driver,
                            'detection_method': 'nvidia-smi'
                        }
                        gpus.append(gpu_info)
            
            print(f"   ✅ nvidia-smi found {len(gpus)} NVIDIA GPU(s)")
        else:
            print("   ⚪ nvidia-smi not available or no NVIDIA GPUs")
    
    except FileNotFoundError:
        print("   ⚪ nvidia-smi not found - no NVIDIA drivers installed")
    except Exception as e:
        print(f"   ⚠️ NVIDIA detection error: {e}")
    
    return gpus

def check_amd_support():
    """Check for AMD GPU support"""
    gpus = []
    
    # AMD doesn't have a universal command-line tool like nvidia-smi
    # We'll rely on WMIC results or registry checks
    
    print("   ℹ️ AMD detection relies on WMIC results above")
    print("   💡 For detailed AMD GPU info, install AMD drivers and check AMD Software")
    
    return gpus

def assess_ai_capability(vendor, memory_gb):
    """Assess AI capability based on vendor and memory"""
    if memory_gb >= 24:
        return "Excellent - Can run large models (LLaMA-30B+)"
    elif memory_gb >= 16:
        return "Very Good - Can run medium models (LLaMA-13B)"
    elif memory_gb >= 8:
        return "Good - Can run small models (LLaMA-7B)"
    elif memory_gb >= 4:
        return "Limited - Basic AI tasks only"
    else:
        return "Poor - Insufficient for modern AI"

if __name__ == "__main__":
    detect_gpus_simple()
