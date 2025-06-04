#!/usr/bin/env python3
"""
DirectML Driver Compatibility Checker
Checks for AMD Adrenalin Edition 23.40.27.06 DirectML requirements and compatibility
"""

import subprocess
import json
import logging
import re
import sys
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DirectMLCompatibilityChecker:
    """
    Checks DirectML driver compatibility and virtual environment requirements
    """
    
    def __init__(self):
        self.required_driver_version = "31.0.24027.6006"
        self.required_adrenalin_version = "23.40.27.06"
        
    def check_amd_driver_compatibility(self) -> Dict[str, Any]:
        """
        Check if the current AMD driver supports DirectML properly
        """
        compatibility_info = {
            'directml_compatible': False,
            'driver_version': None,
            'adrenalin_version': None,
            'virtual_env_supported': False,
            'installation_method': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check installed AMD driver version
            ps_command = '''
            Get-WmiObject Win32_VideoController | 
            Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*"} | 
            Select-Object DriverVersion, DriverDate | 
            ConvertTo-Json
            '''
            
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                driver_data = json.loads(result.stdout)
                
                if isinstance(driver_data, dict):
                    driver_data = [driver_data]
                
                for driver in driver_data:
                    driver_version = driver.get('DriverVersion', '')
                    compatibility_info['driver_version'] = driver_version
                    
                    # Check if driver version matches DirectML requirements
                    if driver_version == self.required_driver_version:
                        compatibility_info['directml_compatible'] = True
                        compatibility_info['adrenalin_version'] = self.required_adrenalin_version
                    elif driver_version:
                        # Check if it's a newer compatible version
                        if self._compare_driver_versions(driver_version, self.required_driver_version) >= 0:
                            compatibility_info['directml_compatible'] = True
                    
                    break  # Use first AMD driver found
                    
        except Exception as e:
            logger.error(f"Failed to check AMD driver version: {e}")
            compatibility_info['issues'].append(f"Driver check failed: {e}")
        
        # Determine virtual environment compatibility
        if compatibility_info['directml_compatible']:
            venv_compatibility = self._check_virtual_env_compatibility()
            compatibility_info.update(venv_compatibility)
        else:
            compatibility_info['issues'].append("DirectML driver not found or incompatible")
            compatibility_info['recommendations'].append("Install AMD Adrenalin Edition 23.40.27.06 for DirectML")
        
        return compatibility_info
    
    def _compare_driver_versions(self, version1: str, version2: str) -> int:
        """
        Compare two driver version strings
        Returns: 1 if version1 > version2, 0 if equal, -1 if version1 < version2
        """
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad with zeros to make equal length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] > v2_parts[i]:
                    return 1
                elif v1_parts[i] < v2_parts[i]:
                    return -1
            
            return 0
            
        except ValueError:
            return 0  # Unable to compare
    
    def _check_virtual_env_compatibility(self) -> Dict[str, Any]:
        """
        Check if DirectML works properly in virtual environments
        """
        venv_info = {
            'virtual_env_supported': False,
            'installation_method': 'system_wide_recommended',
            'issues': [],
            'recommendations': []
        }
        
        # DirectML with AMD drivers has known issues with virtual environments
        # The driver needs system-wide access to GPU resources
        
        venv_info['issues'].extend([
            "DirectML requires system-wide driver access",
            "Virtual environments may isolate driver dependencies",
            "AMD Adrenalin drivers expect global system registration"
        ])
        
        venv_info['recommendations'].extend([
            "Install Python packages system-wide for AMD DirectML",
            "Use conda environments instead of venv (better driver compatibility)",
            "Ensure Windows PATH includes AMD driver directories",
            "Use separate Python installations rather than virtual environments"
        ])
        
        # Check if we're currently in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_info['issues'].append("Currently running in virtual environment - may cause DirectML issues")
            venv_info['installation_method'] = 'current_venv_risky'
        else:
            venv_info['installation_method'] = 'system_wide_current'
            venv_info['virtual_env_supported'] = True  # System installation is OK
        
        return venv_info
    
    def check_directml_installation(self) -> Dict[str, Any]:
        """
        Check if DirectML packages are properly installed and accessible
        """
        installation_info = {
            'torch_directml_available': False,
            'tensorflow_directml_available': False,
            'onnxruntime_directml_available': False,
            'installation_location': 'unknown',
            'issues': [],
            'working_packages': []
        }
        
        # Check torch-directml
        try:
            import torch_directml
            installation_info['torch_directml_available'] = True
            installation_info['working_packages'].append('torch-directml')
            
            # Test device count
            device_count = torch_directml.device_count()
            installation_info['torch_directml_devices'] = device_count
            
        except ImportError:
            installation_info['issues'].append("torch-directml not installed")
        except Exception as e:
            installation_info['issues'].append(f"torch-directml error: {e}")
        
        # Check tensorflow-directml
        try:
            import tensorflow_directml
            installation_info['tensorflow_directml_available'] = True
            installation_info['working_packages'].append('tensorflow-directml')
            
        except ImportError:
            installation_info['issues'].append("tensorflow-directml not installed")
        except Exception as e:
            installation_info['issues'].append(f"tensorflow-directml error: {e}")
        
        # Check onnxruntime-directml
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                installation_info['onnxruntime_directml_available'] = True
                installation_info['working_packages'].append('onnxruntime-directml')
            else:
                installation_info['issues'].append("onnxruntime DirectML provider not available")
                
        except ImportError:
            installation_info['issues'].append("onnxruntime not installed")
        except Exception as e:
            installation_info['issues'].append(f"onnxruntime error: {e}")
        
        # Determine installation location
        if installation_info['working_packages']:
            installation_info['installation_location'] = sys.prefix
        
        return installation_info
    
    def get_recommended_installation_strategy(self, gpu_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get recommended installation strategy based on GPU setup and DirectML compatibility
        """
        strategy = {
            'approach': 'mixed_strategy',
            'amd_installation_method': 'system_wide',
            'nvidia_installation_method': 'virtual_env',
            'steps': [],
            'warnings': [],
            'environment_setup': {}
        }
        
        nvidia_gpus = [gpu for gpu in gpu_info if gpu.get('vendor', '').upper() == 'NVIDIA']
        amd_gpus = [gpu for gpu in gpu_info if gpu.get('vendor', '').upper() == 'AMD']
        
        compatibility = self.check_amd_driver_compatibility()
        
        if amd_gpus and compatibility['directml_compatible']:
            # AMD GPUs with DirectML - recommend system-wide installation
            strategy['steps'].extend([
                "1. Install Python packages system-wide for AMD DirectML compatibility",
                "2. Ensure AMD Adrenalin Edition 23.40.27.06 is installed",
                "3. Install DirectML packages using system Python:",
                "   - pip install torch torchvision torchaudio",
                "   - pip install torch-directml",
                "   - pip install tensorflow-directml",
                "   - pip install onnxruntime-directml"
            ])
            
            if nvidia_gpus:
                # Mixed environment - special handling
                strategy['approach'] = 'hybrid_installation'
                strategy['steps'].extend([
                    "4. For NVIDIA GPUs, create separate virtual environment:",
                    "   - python -m venv nvidia_gpu_env",
                    "   - nvidia_gpu_env\\Scripts\\activate",
                    "   - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                    "   - pip install tensorflow[and-cuda]",
                    "5. Use device selection code to target appropriate GPU vendor"
                ])
                
                strategy['warnings'].extend([
                    "Mixed AMD/NVIDIA setup requires careful package management",
                    "AMD DirectML must be installed system-wide",
                    "NVIDIA CUDA packages can use virtual environments",
                    "Use explicit device selection in your code"
                ])
            
            strategy['environment_setup'] = {
                'amd_directml': {
                    'location': 'system_wide',
                    'python_path': sys.executable,
                    'packages': [
                        'torch', 'torchvision', 'torchaudio',
                        'torch-directml',
                        'tensorflow-directml',
                        'onnxruntime-directml'
                    ],
                    'activation': 'Use system Python directly'
                }
            }
            
            if nvidia_gpus:
                strategy['environment_setup']['nvidia_cuda'] = {
                    'location': 'virtual_environment',
                    'path': 'gpu_environments/nvidia_gpu_env',
                    'packages': [
                        'torch+cu118', 'torchvision+cu118', 'torchaudio+cu118',
                        'tensorflow[and-cuda]',
                        'onnxruntime-gpu'
                    ],
                    'activation': 'gpu_environments\\nvidia_gpu_env\\Scripts\\activate'
                }
        
        elif amd_gpus and not compatibility['directml_compatible']:
            strategy['steps'].extend([
                "1. ⚠️ AMD DirectML driver not compatible",
                "2. Install AMD Adrenalin Edition 23.40.27.06 for DirectML",
                "3. Download from: https://drivers.amd.com/drivers/amd-software-adrenalin-edition-23.40.27.06-win10-win11-may-rdna.exe",
                "4. Run AMD Cleanup Utility before installation",
                "5. Restart system after driver installation",
                "6. Re-run this setup"
            ])
            
            strategy['warnings'].append("AMD driver update required for DirectML support")
        
        return strategy

def main():
    """Demonstrate DirectML compatibility checking"""
    checker = DirectMLCompatibilityChecker()
    
    print("🔍 DirectML Compatibility Check")
    print("=" * 40)
    
    # Check driver compatibility
    driver_compat = checker.check_amd_driver_compatibility()
    print(f"DirectML Compatible: {driver_compat['directml_compatible']}")
    print(f"Driver Version: {driver_compat['driver_version']}")
    print(f"Virtual Env Supported: {driver_compat['virtual_env_supported']}")
    
    if driver_compat['issues']:
        print("Issues found:")
        for issue in driver_compat['issues']:
            print(f"  - {issue}")
    
    if driver_compat['recommendations']:
        print("Recommendations:")
        for rec in driver_compat['recommendations']:
            print(f"  - {rec}")
    
    # Check current installation
    installation = checker.check_directml_installation()
    print(f"\nDirectML Packages:")
    print(f"  torch-directml: {installation['torch_directml_available']}")
    print(f"  tensorflow-directml: {installation['tensorflow_directml_available']}")
    print(f"  onnxruntime-directml: {installation['onnxruntime_directml_available']}")
    
    # Save results
    with open('directml_compatibility_check.json', 'w') as f:
        json.dump({
            'driver_compatibility': driver_compat,
            'installation_status': installation
        }, f, indent=2)
    
    print("\n📄 Results saved to directml_compatibility_check.json")

if __name__ == "__main__":
    main()
