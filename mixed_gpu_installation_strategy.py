#!/usr/bin/env python3
"""
Mixed GPU Environment Installation Strategy
Handles AMD DirectML (system-wide) + NVIDIA CUDA (virtual env) setup
"""

import subprocess
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MixedGPUInstallationStrategy:
    """
    Manages installation strategy for mixed AMD/NVIDIA GPU systems
    """
    
    def __init__(self):
        self.detected_gpus = self.detect_gpus()
        self.has_amd = any(gpu.get('vendor', '').upper() == 'AMD' for gpu in self.detected_gpus)
        self.has_nvidia = any(gpu.get('vendor', '').upper() == 'NVIDIA' for gpu in self.detected_gpus)
        
    def detect_gpus(self) -> List[Dict[str, Any]]:
        """Simple GPU detection for mixed setups"""
        gpus = []
        
        try:
            # Check for AMD GPUs
            ps_command = '''
            Get-WmiObject Win32_VideoController | 
            Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*"} | 
            Select-Object Name, DriverVersion | ConvertTo-Json
            '''
            
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                amd_data = json.loads(result.stdout)
                if isinstance(amd_data, dict):
                    amd_data = [amd_data]
                
                for gpu in amd_data:
                    gpus.append({
                        'vendor': 'AMD',
                        'name': gpu.get('Name', 'Unknown AMD GPU'),
                        'driver_version': gpu.get('DriverVersion', ''),
                        'directml_compatible': gpu.get('DriverVersion') == '31.0.24027.6006'
                    })
                    
        except Exception as e:
            logger.warning(f"AMD GPU detection failed: {e}")
        
        try:
            # Check for NVIDIA GPUs using nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            gpus.append({
                                'vendor': 'NVIDIA',
                                'name': parts[0].strip(),
                                'driver_version': parts[1].strip(),
                                'cuda_compatible': True
                            })
                            
        except Exception as e:
            logger.warning(f"NVIDIA GPU detection failed: {e}")
        
        return gpus
    
    def create_mixed_installation_plan(self) -> Dict[str, Any]:
        """
        Create installation plan for mixed AMD/NVIDIA setup
        """
        plan = {
            'strategy': 'mixed_gpu_hybrid',
            'detected_gpus': self.detected_gpus,
            'has_amd': self.has_amd,
            'has_nvidia': self.has_nvidia,
            'amd_installation': {},
            'nvidia_installation': {},
            'installation_order': [],
            'warnings': [],
            'verification_steps': []
        }
        
        if self.has_amd:
            # AMD DirectML - MUST be system-wide
            plan['amd_installation'] = {
                'method': 'system_wide_required',
                'reason': 'DirectML driver integration requires system-wide Python',
                'packages': [
                    'torch',
                    'torchvision',
                    'torchaudio',
                    'torch-directml',
                    'tensorflow-directml',
                    'onnxruntime-directml',
                    'numpy>=1.21.0',
                    'pillow>=8.0.0'
                ],
                'commands': [
                    'python -m pip install --upgrade pip',
                    'python -m pip install torch torchvision torchaudio',
                    'python -m pip install torch-directml',
                    'python -m pip install tensorflow-directml',
                    'python -m pip install onnxruntime-directml',
                    'python -m pip install numpy pillow opencv-python'
                ],
                'requirements': [
                    'Exit any virtual environments',
                    'Use system Python only',
                    'AMD Adrenalin Edition 23.40.27.06 required'
                ],
                'verification': [
                    'python -c "import torch_directml; print(f\'DirectML devices: {torch_directml.device_count()}\')"',
                    'python -c "import tensorflow as tf; print(f\'TF GPUs: {len(tf.config.list_physical_devices(\'GPU\'))}\')"'
                ]
            }
            
            plan['installation_order'].append('amd_system_wide')
            plan['warnings'].append('AMD DirectML requires system-wide installation - no virtual environments')
        
        if self.has_nvidia:
            # NVIDIA CUDA - Can use virtual environment
            plan['nvidia_installation'] = {
                'method': 'virtual_environment_supported',
                'reason': 'CUDA has good virtual environment compatibility',
                'environment_path': 'gpu_environments/nvidia_cuda_env',
                'packages': [
                    'torch+cu118',
                    'torchvision+cu118', 
                    'torchaudio+cu118',
                    'tensorflow[and-cuda]',
                    'onnxruntime-gpu',
                    'nvidia-ml-py3',
                    'cupy-cuda11x'
                ],
                'commands': [
                    'python -m venv gpu_environments/nvidia_cuda_env',
                    'gpu_environments/nvidia_cuda_env/Scripts/activate',
                    'pip install --upgrade pip',
                    'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
                    'pip install tensorflow[and-cuda]',
                    'pip install onnxruntime-gpu nvidia-ml-py3'
                ],
                'verification': [
                    'python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"',
                    'python -c "import tensorflow as tf; print(f\'TF GPUs: {len(tf.config.list_physical_devices(\'GPU\'))}\')"'
                ]
            }
            
            plan['installation_order'].append('nvidia_virtual_env')
        
        # Mixed setup specific configurations
        if self.has_amd and self.has_nvidia:
            plan['mixed_setup_config'] = {
                'device_selection_strategy': 'explicit_vendor_targeting',
                'amd_usage': 'Use system Python + DirectML packages',
                'nvidia_usage': 'Activate nvidia_cuda_env + CUDA packages',
                'code_examples': [
                    '# For AMD DirectML usage:',
                    'import torch_directml',
                    'amd_device = torch_directml.device()',
                    '',
                    '# For NVIDIA CUDA usage (in nvidia_cuda_env):',
                    'import torch',
                    'nvidia_device = torch.device("cuda:0")',
                ],
                'environment_switching': [
                    'AMD work: Use system Python directly',
                    'NVIDIA work: Activate gpu_environments/nvidia_cuda_env'
                ]
            }
            
            plan['warnings'].extend([
                'Mixed setup requires careful environment management',
                'AMD work: system Python only',
                'NVIDIA work: activate virtual environment'
            ])
        
        return plan
    
    def execute_installation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the mixed GPU installation plan
        """
        execution_results = {
            'start_time': datetime.now().isoformat(),
            'amd_results': {},
            'nvidia_results': {},
            'overall_success': False,
            'errors': [],
            'warnings': []
        }
        
        print("🚀 Executing Mixed GPU Installation Plan")
        print("=" * 50)
        
        # Check current environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv and plan['has_amd']:
            execution_results['errors'].append(
                "❌ CRITICAL: In virtual environment but AMD DirectML requires system Python"
            )
            execution_results['warnings'].append(
                "Exit virtual environment before installing AMD DirectML packages"
            )
            print("❌ CRITICAL: Exit virtual environment for AMD DirectML installation")
            return execution_results
        
        # AMD Installation (if present)
        if plan['has_amd']:
            print("\\n📦 Installing AMD DirectML (System-Wide)")
            print("-" * 40)
            
            amd_success = True
            for cmd in plan['amd_installation']['commands']:
                try:
                    print(f"Running: {cmd}")
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=600
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ Success: {cmd.split()[-1] if cmd.split() else 'command'}")
                    else:
                        print(f"❌ Failed: {cmd}")
                        print(f"Error: {result.stderr[:200]}...")
                        amd_success = False
                        execution_results['errors'].append(f"AMD command failed: {cmd}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⏰ Timeout: {cmd}")
                    amd_success = False
                    execution_results['errors'].append(f"AMD command timeout: {cmd}")
                except Exception as e:
                    print(f"❌ Error: {cmd} - {e}")
                    amd_success = False
                    execution_results['errors'].append(f"AMD command error: {cmd} - {e}")
            
            execution_results['amd_results']['success'] = amd_success
            
            # Verify AMD installation
            if amd_success:
                print("\\n🧪 Verifying AMD DirectML installation...")
                for verify_cmd in plan['amd_installation']['verification']:
                    try:
                        result = subprocess.run(
                            verify_cmd, shell=True, capture_output=True, text=True, timeout=30
                        )
                        if result.returncode == 0:
                            print(f"✅ {result.stdout.strip()}")
                        else:
                            print(f"⚠️ Verification issue: {result.stderr[:100]}")
                    except Exception as e:
                        print(f"⚠️ Verification error: {e}")
        
        # NVIDIA Installation (if present)
        if plan['has_nvidia']:
            print("\\n📦 Installing NVIDIA CUDA (Virtual Environment)")
            print("-" * 40)
            
            nvidia_success = True
            for cmd in plan['nvidia_installation']['commands']:
                try:
                    print(f"Running: {cmd}")
                    
                    # Handle virtual environment activation specially
                    if 'activate' in cmd:
                        print(f"📝 Note: {cmd}")
                        print("    (Virtual environment setup - run manually if needed)")
                        continue
                    
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=600
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ Success: {cmd.split()[-1] if cmd.split() else 'command'}")
                    else:
                        print(f"❌ Failed: {cmd}")
                        print(f"Error: {result.stderr[:200]}...")
                        nvidia_success = False
                        execution_results['errors'].append(f"NVIDIA command failed: {cmd}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⏰ Timeout: {cmd}")
                    nvidia_success = False
                except Exception as e:
                    print(f"❌ Error: {cmd} - {e}")
                    nvidia_success = False
            
            execution_results['nvidia_results']['success'] = nvidia_success
        
        # Overall success
        amd_ok = execution_results['amd_results'].get('success', True)  # True if no AMD
        nvidia_ok = execution_results['nvidia_results'].get('success', True)  # True if no NVIDIA
        
        execution_results['overall_success'] = amd_ok and nvidia_ok
        execution_results['end_time'] = datetime.now().isoformat()
        
        return execution_results
    
    def run_mixed_gpu_setup(self) -> Dict[str, Any]:
        """
        Run complete mixed GPU setup process
        """
        print("🔍 Mixed GPU Environment Setup")
        print("=" * 40)
        
        # Display detected GPUs
        print(f"Detected GPUs: {len(self.detected_gpus)}")
        for gpu in self.detected_gpus:
            vendor = gpu.get('vendor', 'Unknown')
            name = gpu.get('name', 'Unknown')
            print(f"  {vendor}: {name}")
        
        # Create installation plan
        plan = self.create_mixed_installation_plan()
        
        # Display plan summary
        print(f"\\nInstallation Strategy: {plan['strategy']}")
        if plan['warnings']:
            print("⚠️ Warnings:")
            for warning in plan['warnings']:
                print(f"   - {warning}")
        
        # Execute plan
        results = self.execute_installation_plan(plan)
        
        # Final summary
        print("\\n" + "=" * 50)
        print("Mixed GPU Setup Results:")
        print(f"Overall Success: {'✅' if results['overall_success'] else '❌'}")
        
        if results['errors']:
            print("\\nErrors:")
            for error in results['errors']:
                print(f"   {error}")
        
        # Save results
        with open('mixed_gpu_setup_results.json', 'w') as f:
            json.dump({
                'plan': plan,
                'execution_results': results
            }, f, indent=2)
        
        print("\\n📄 Results saved to: mixed_gpu_setup_results.json")
        
        return {
            'plan': plan,
            'execution_results': results
        }

def main():
    """Run mixed GPU installation strategy"""
    installer = MixedGPUInstallationStrategy()
    results = installer.run_mixed_gpu_setup()
    
    if results['execution_results']['overall_success']:
        print("\\n🎉 Mixed GPU setup completed successfully!")
        
        # Provide usage guidance
        if installer.has_amd and installer.has_nvidia:
            print("\\n💡 Usage Guide:")
            print("   AMD DirectML: Use system Python directly")
            print("   NVIDIA CUDA: Activate gpu_environments/nvidia_cuda_env")
        elif installer.has_amd:
            print("\\n💡 AMD DirectML ready - use system Python")
        elif installer.has_nvidia:
            print("\\n💡 NVIDIA CUDA ready - use virtual environment")
    else:
        print("\\n⚠️ Mixed GPU setup had issues - check results file")

if __name__ == "__main__":
    main()
