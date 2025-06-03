#!/usr/bin/env python3
"""
Enhanced AMD GPU Detection for DirectML
Specifically targets AMD Adrenalin 23 installations
"""

import subprocess
import json
import re
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_amd_gpus_powershell() -> List[Dict[str, Any]]:
    """
    Use PowerShell to detect AMD GPUs via WMI
    This method works even without Python WMI packages
    """
    amd_gpus = []
    
    try:
        # PowerShell command to get video controller information
        ps_command = '''
        Get-WmiObject Win32_VideoController | 
        Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*"} | 
        Select-Object Name, AdapterRAM, PNPDeviceID, Status, DriverVersion | 
        ConvertTo-Json
        '''
        
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            try:
                # Parse JSON output
                gpu_data = json.loads(result.stdout)
                
                # Handle both single GPU (dict) and multiple GPUs (list)
                if isinstance(gpu_data, dict):
                    gpu_data = [gpu_data]
                
                for gpu in gpu_data:
                    if gpu.get('Name'):
                        # Convert memory from bytes to GB
                        memory_bytes = gpu.get('AdapterRAM', 0)
                        memory_gb = memory_bytes / (1024**3) if memory_bytes else 0
                        
                        gpu_info = {
                            'name': gpu['Name'],
                            'memory_gb': round(memory_gb, 2),
                            'memory_mb': round(memory_gb * 1024, 0),
                            'driver_version': gpu.get('DriverVersion', 'Unknown'),
                            'status': gpu.get('Status', 'Unknown'),
                            'pnp_device_id': gpu.get('PNPDeviceID', 'Unknown'),
                            'vendor': 'AMD',
                            'type': 'GPU',
                            'compute_capability': determine_compute_capability(gpu['Name']),
                            'directml_compatible': is_directml_compatible(gpu['Name']),
                            'detection_method': 'powershell_wmi'
                        }
                        
                        amd_gpus.append(gpu_info)
                        logger.info(f"Detected AMD GPU: {gpu['Name']} ({memory_gb:.2f}GB)")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse PowerShell JSON output: {e}")
                
    except subprocess.TimeoutExpired:
        logger.error("PowerShell GPU detection timed out")
    except Exception as e:
        logger.error(f"PowerShell GPU detection failed: {e}")
    
    return amd_gpus

def detect_amd_gpus_dxdiag() -> List[Dict[str, Any]]:
    """
    Use DirectX diagnostics to detect AMD GPUs
    """
    amd_gpus = []
    
    try:
        # Run dxdiag and capture output
        result = subprocess.run(
            ["dxdiag", "/t", "temp_dxdiag.txt"],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0:
            try:
                with open("temp_dxdiag.txt", "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Parse AMD GPU information from dxdiag output
                gpu_pattern = r"Card name:\s*(.*AMD.*Radeon.*)"
                memory_pattern = r"Display Memory:\s*(\d+)\s*MB"
                chip_pattern = r"Chip type:\s*(.*AMD.*)"
                
                gpu_matches = re.finditer(gpu_pattern, content, re.IGNORECASE)
                
                for match in gpu_matches:
                    gpu_name = match.group(1).strip()
                    
                    # Find memory info near this GPU entry
                    start_pos = match.start()
                    section = content[start_pos:start_pos + 2000]  # Look in next 2000 chars
                    
                    memory_match = re.search(memory_pattern, section)
                    memory_mb = int(memory_match.group(1)) if memory_match else 0
                    memory_gb = memory_mb / 1024 if memory_mb else 0
                    
                    gpu_info = {
                        'name': gpu_name,
                        'memory_gb': round(memory_gb, 2),
                        'memory_mb': memory_mb,
                        'vendor': 'AMD',
                        'type': 'GPU',
                        'compute_capability': determine_compute_capability(gpu_name),
                        'directml_compatible': is_directml_compatible(gpu_name),
                        'detection_method': 'dxdiag'
                    }
                    
                    amd_gpus.append(gpu_info)
                    logger.info(f"Detected AMD GPU via DxDiag: {gpu_name} ({memory_gb:.2f}GB)")
                
                # Clean up temp file
                import os
                try:
                    os.remove("temp_dxdiag.txt")
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to parse dxdiag output: {e}")
                
    except subprocess.TimeoutExpired:
        logger.error("DxDiag timed out")
    except Exception as e:
        logger.error(f"DxDiag detection failed: {e}")
    
    return amd_gpus

def determine_compute_capability(gpu_name: str) -> str:
    """
    Determine compute capability based on GPU name
    """
    gpu_name_lower = gpu_name.lower()
    
    # RDNA2 architecture (RX 6000 series)
    if any(model in gpu_name_lower for model in ['6800', '6900', '6700', '6600', '6500', '6400']):
        return "RDNA2"
    
    # RDNA3 architecture (RX 7000 series)
    if any(model in gpu_name_lower for model in ['7800', '7900', '7700', '7600']):
        return "RDNA3"
    
    # RDNA architecture (RX 5000 series)
    if any(model in gpu_name_lower for model in ['5700', '5600', '5500']):
        return "RDNA"
    
    # Vega architecture
    if 'vega' in gpu_name_lower:
        return "Vega"
    
    # Polaris architecture
    if any(model in gpu_name_lower for model in ['580', '570', '560', '550']):
        return "Polaris"
    
    return "Unknown"

def is_directml_compatible(gpu_name: str) -> bool:
    """
    Check if GPU is DirectML compatible
    """
    gpu_name_lower = gpu_name.lower()
    
    # DirectML supports RDNA2 and newer
    rdna2_compatible = any(model in gpu_name_lower for model in [
        '6800', '6900', '6700', '6600', '6500', '6400',  # RDNA2
        '7800', '7900', '7700', '7600'  # RDNA3
    ])
    
    # Also supports some Vega and newer integrated GPUs
    vega_compatible = 'vega' in gpu_name_lower
    
    return rdna2_compatible or vega_compatible

def run_enhanced_detection():
    """
    Run all detection methods and combine results
    """
    print("🚀 Enhanced AMD GPU Detection")
    print("=" * 50)
    
    all_gpus = []
    
    # Method 1: PowerShell WMI
    print("🔍 Method 1: PowerShell WMI Detection")
    ps_gpus = detect_amd_gpus_powershell()
    if ps_gpus:
        print(f"✅ Found {len(ps_gpus)} AMD GPU(s) via PowerShell")
        all_gpus.extend(ps_gpus)
    else:
        print("❌ No AMD GPUs found via PowerShell")
    
    # Method 2: DirectX Diagnostics
    print("\n🔍 Method 2: DirectX Diagnostics Detection")
    dx_gpus = detect_amd_gpus_dxdiag()
    if dx_gpus:
        print(f"✅ Found {len(dx_gpus)} AMD GPU(s) via DxDiag")
        # Only add if not already found
        for gpu in dx_gpus:
            if not any(existing['name'] == gpu['name'] for existing in all_gpus):
                all_gpus.append(gpu)
    else:
        print("❌ No AMD GPUs found via DxDiag")
    
    # Summary
    print(f"\n📊 Summary: Found {len(all_gpus)} unique AMD GPU(s)")
    
    for i, gpu in enumerate(all_gpus, 1):
        print(f"\n🎯 GPU {i}: {gpu['name']}")
        print(f"   Memory: {gpu['memory_gb']:.2f} GB")
        print(f"   Architecture: {gpu['compute_capability']}")
        print(f"   DirectML Compatible: {'✅' if gpu['directml_compatible'] else '❌'}")
        print(f"   Detection Method: {gpu['detection_method']}")
    
    # Save results
    results = {
        'total_gpus': len(all_gpus),
        'gpus': all_gpus,
        'directml_compatible_count': sum(1 for gpu in all_gpus if gpu['directml_compatible'])
    }
    
    with open('enhanced_amd_detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: enhanced_amd_detection_results.json")
    
    return all_gpus

if __name__ == "__main__":
    run_enhanced_detection()
