#!/usr/bin/env python3
"""
DirectML System-Wide Installation Plan
Handles proper DirectML installation that respects system-wide requirements
"""

import subprocess
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectMLInstallationPlan:
    """
    Creates proper installation plans that respect DirectML system-wide requirements
    """
    
    def __init__(self):
        self.detected_gpus = self._detect_gpus()
        self.has_amd = any(gpu.get('vendor', '').upper() == 'AMD' for gpu in self.detected_gpus)
        self.has_nvidia = any(gpu.get('vendor', '').upper() == 'NVIDIA' for gpu in self.detected_gpus)
        
    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Simple GPU detection for installation planning"""
        gpus = []
        
        try:
            # AMD GPU detection
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
                        'directml_compatible': True
                    })
                    
        except Exception as e:
            logger.warning(f"AMD GPU detection failed: {e}")
        
        try:
            # NVIDIA GPU detection
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
    
    def create_directml_system_wide_plan(self) -> Dict[str, Any]:
        """
        Create installation plan that properly handles DirectML system-wide requirements
        """
        plan = {
            'strategy': 'directml_system_wide_required',
            'detected_gpus': self.detected_gpus,
            'has_amd': self.has_amd,
            'has_nvidia': self.has_nvidia,
            'environments': {},
            'critical_warnings': [],
            'installation_order': [],
            'verification_steps': []
        }
        
        # Critical DirectML requirements
        if self.has_amd:
            plan['critical_warnings'].extend([
                "❌ CRITICAL: DirectML MUST use system-wide Python installation",
                "❌ CRITICAL: Virtual environments WILL BREAK DirectML functionality", 
                "❌ CRITICAL: AMD Adrenalin Edition 23.40.27.06 driver required",
                "⚠️ DirectML driver integration requires global Python access"
            ])
            
            # AMD DirectML environment (system-wide ONLY)
            plan['environments']['amd_directml'] = {
                'name': 'system_wide_directml',
                'installation_method': 'system_wide_mandatory',
                'python_requirement': 'system_python_only',
                'virtual_env_compatible': False,
                'packages': [
                    'torch>=2.0.0',
                    'torchvision>=0.15.0',
                    'torchaudio>=2.0.0',
                    'torch-directml>=0.2.0',
                    'onnxruntime-directml>=1.15.0',
                    'numpy>=1.21.0',
                    'pillow>=8.0.0'
                ],
                'installation_commands': [
                    'python -m pip install --upgrade pip',
                    'python -m pip install torch torchvision torchaudio',
                    'python -m pip install torch-directml',
                    'python -m pip install onnxruntime-directml',
                    'python -m pip install numpy pillow opencv-python'
                ],
                'pre_installation_checks': [
                    'Verify not in virtual environment: deactivate',
                    'Check system Python: python --version',
                    'Verify AMD driver: AMD Adrenalin Edition 23.40.27.06'
                ],
                'verification_commands': [
                    'python -c "import torch_directml; print(f\'DirectML devices: {torch_directml.device_count()}\')"',
                    'python -c "import torch_directml; device = torch_directml.device(); print(f\'DirectML device: {device}\')"'
                ],
                'environment_variables': {
                    'TORCH_DIRECTML_DEVICE': '0',
                    'DML_VISIBLE_DEVICES': 'all'
                },
                'usage_pattern': 'direct_system_python_execution'
            }
            
            plan['installation_order'].append('check_virtual_environment_exit')
            plan['installation_order'].append('install_amd_directml_system_wide')
        
        # NVIDIA CUDA environment (virtual env compatible)
        if self.has_nvidia:
            plan['environments']['nvidia_cuda'] = {
                'name': 'cuda_virtual_env',
                'installation_method': 'virtual_environment_supported',
                'python_requirement': 'system_or_virtual_env',
                'virtual_env_compatible': True,
                'environment_path': 'gpu_environments/nvidia_cuda_env',
                'packages': [
                    'torch+cu118',
                    'torchvision+cu118',
                    'torchaudio+cu118',
                    'tensorflow[and-cuda]',
                    'onnxruntime-gpu',
                    'nvidia-ml-py3'
                ],
                'installation_commands': [
                    'python -m venv gpu_environments/nvidia_cuda_env',
                    'gpu_environments/nvidia_cuda_env/Scripts/activate.bat',
                    'pip install --upgrade pip',
                    'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
                    'pip install tensorflow[and-cuda]',
                    'pip install onnxruntime-gpu nvidia-ml-py3'
                ],
                'verification_commands': [
                    'python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"',
                    'python -c "import torch; print(f\'CUDA devices: {torch.cuda.device_count()}\')"'
                ],
                'usage_pattern': 'activate_virtual_environment_first'
            }
            
            plan['installation_order'].append('create_nvidia_virtual_env')
            plan['installation_order'].append('install_nvidia_cuda_packages')
        
        # Mixed environment usage guidance
        if self.has_amd and self.has_nvidia:
            plan['mixed_usage_guide'] = {
                'amd_workflow': [
                    '1. Ensure you are NOT in any virtual environment',
                    '2. Use system Python directly',
                    '3. import torch_directml',
                    '4. device = torch_directml.device()',
                    '5. tensor.to(device)'
                ],
                'nvidia_workflow': [
                    '1. Activate CUDA virtual environment',
                    '2. gpu_environments/nvidia_cuda_env/Scripts/activate',
                    '3. import torch',
                    '4. device = torch.device("cuda:0")',
                    '5. tensor.to(device)'
                ],
                'environment_switching': {
                    'for_amd_work': 'deactivate (exit any virtual env)',
                    'for_nvidia_work': 'activate gpu_environments/nvidia_cuda_env/Scripts/activate'
                }
            }
        
        # Verification steps
        plan['verification_steps'] = [
            'Test DirectML device enumeration (system Python)',
            'Test CUDA device enumeration (virtual env)',
            'Verify no package conflicts',
            'Test tensor operations on each device type'
        ]
        
        return plan
    
    def generate_installation_script(self, plan: Dict[str, Any]) -> str:
        """
        Generate batch script for proper DirectML installation
        """
        script_content = '''@echo off
echo DirectML System-Wide Installation Script
echo ==========================================
echo.

REM Critical DirectML Requirements Check
echo [STEP 1] Checking DirectML Requirements...
echo.
echo CRITICAL REQUIREMENTS:
echo 1. AMD Adrenalin Edition 23.40.27.06 driver installed
echo 2. NO virtual environments active
echo 3. System Python installation available
echo.

REM Check if in virtual environment
if defined VIRTUAL_ENV (
    echo ERROR: Virtual environment detected: %VIRTUAL_ENV%
    echo DirectML REQUIRES system-wide Python installation
    echo Please run: deactivate
    echo Then re-run this script
    pause
    exit /b 1
)

echo [STEP 2] Installing AMD DirectML packages (system-wide)...
echo.
python -m pip install --upgrade pip
if errorlevel 1 goto error

python -m pip install torch torchvision torchaudio
if errorlevel 1 goto error

python -m pip install torch-directml
if errorlevel 1 goto error

python -m pip install onnxruntime-directml
if errorlevel 1 goto error

python -m pip install numpy pillow opencv-python
if errorlevel 1 goto error

echo.
echo [STEP 3] Verifying DirectML installation...
echo.
python -c "import torch_directml; print(f'DirectML devices: {torch_directml.device_count()}')"
if errorlevel 1 goto error

python -c "import torch_directml; device = torch_directml.device(); print(f'DirectML device: {device}')"
if errorlevel 1 goto error

echo.
echo ==========================================
echo DirectML Installation Completed Successfully!
echo ==========================================
echo.
echo IMPORTANT USAGE NOTES:
echo 1. Always use system Python for DirectML workloads
echo 2. NEVER activate virtual environments for AMD GPU work
echo 3. Virtual environments will break DirectML functionality
echo.
echo Test DirectML with:
echo python -c "import torch_directml; print('DirectML ready!')"
echo.
pause
exit /b 0

:error
echo.
echo ERROR: Installation failed!
echo Check the error messages above
echo.
pause
exit /b 1
'''
        
        return script_content
    
    def run_installation_analysis(self) -> Dict[str, Any]:
        """
        Run complete DirectML installation analysis
        """
        print("🔍 DirectML System-Wide Installation Analysis")
        print("=" * 50)
        
        # Display GPU detection
        print(f"Detected GPUs: {len(self.detected_gpus)}")
        for gpu in self.detected_gpus:
            vendor = gpu.get('vendor', 'Unknown')
            name = gpu.get('name', 'Unknown')
            print(f"  {vendor}: {name}")
        
        # Create installation plan
        plan = self.create_directml_system_wide_plan()
        
        # Display critical warnings
        if plan['critical_warnings']:
            print("\\n⚠️ CRITICAL WARNINGS:")
            for warning in plan['critical_warnings']:
                print(f"  {warning}")
        
        # Check current environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        print(f"\\n📊 Environment Status:")
        print(f"  Current Python: {sys.executable}")
        print(f"  In Virtual Env: {'Yes' if in_venv else 'No'}")
        print(f"  DirectML Compatible: {'No' if in_venv else 'Yes'}")
        
        if in_venv and self.has_amd:
            print("\\n❌ CRITICAL ISSUE:")
            print("  Virtual environment detected but AMD GPUs present")
            print("  DirectML will NOT work in virtual environments")
            print("  Solution: deactivate virtual environment")
        
        # Generate installation script
        script_content = self.generate_installation_script(plan)
        script_path = 'install_directml_system_wide.bat'
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"\\n📄 Generated installation script: {script_path}")
        
        # Save plan
        plan_path = 'directml_installation_plan.json'
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, default=str)
        
        print(f"📄 Installation plan saved: {plan_path}")
        
        return {
            'plan': plan,
            'script_generated': script_path,
            'plan_saved': plan_path,
            'environment_compatible': not in_venv,
            'recommendations': self._get_recommendations(plan, in_venv)
        }
    
    def _get_recommendations(self, plan: Dict[str, Any], in_venv: bool) -> List[str]:
        """Get specific recommendations based on current environment"""
        recommendations = []
        
        if self.has_amd:
            if in_venv:
                recommendations.extend([
                    "1. Exit virtual environment: deactivate",
                    "2. Run install_directml_system_wide.bat",
                    "3. Test: python -c \"import torch_directml\""
                ])
            else:
                recommendations.extend([
                    "1. Run install_directml_system_wide.bat",
                    "2. Always use system Python for DirectML",
                    "3. Never activate virtual environments for AMD work"
                ])
        
        if self.has_nvidia:
            recommendations.extend([
                "4. Create NVIDIA virtual environment for CUDA work",
                "5. Use virtual environments freely for NVIDIA workflows"
            ])
        
        return recommendations

def main():
    """Run DirectML installation planning"""
    planner = DirectMLInstallationPlan()
    results = planner.run_installation_analysis()
    
    if results['recommendations']:
        print("\\n💡 Next Steps:")
        for rec in results['recommendations']:
            print(f"  {rec}")
    
    print("\\n✅ DirectML installation analysis complete!")

if __name__ == "__main__":
    main()
