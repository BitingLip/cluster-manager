#!/usr/bin/env python3
"""
AMD DirectML Installation Guide
Handles proper system-wide installation for AMD Adrenalin Edition 23.40.27.06
Virtual environments are NOT recommended for AMD DirectML due to driver integration
"""

import subprocess
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AMDDirectMLInstallationGuide:
    """
    Guides proper AMD DirectML installation for Adrenalin Edition 23.40.27.06
    """
    
    def __init__(self):
        self.required_driver_version = "31.0.24027.6006"
        self.adrenalin_version = "23.40.27.06"
        
    def check_installation_environment(self) -> Dict[str, Any]:
        """
        Check if the current environment is suitable for DirectML installation
        """
        env_check = {
            'current_python': sys.executable,
            'in_virtual_env': False,
            'system_python_available': True,
            'admin_rights': False,
            'amd_driver_present': False,
            'recommendations': [],
            'critical_issues': []
        }
        
        # Check if in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            env_check['in_virtual_env'] = True
            env_check['critical_issues'].append(
                "❌ CRITICAL: Running in virtual environment - DirectML will likely fail"
            )
            env_check['recommendations'].append(
                "Exit virtual environment and use system Python for DirectML installation"
            )
        
        # Check admin rights
        try:
            import ctypes
            env_check['admin_rights'] = ctypes.windll.shell32.IsUserAnAdmin()
        except:
            env_check['admin_rights'] = False
        
        if not env_check['admin_rights']:
            env_check['recommendations'].append(
                "Run as Administrator for best DirectML installation compatibility"
            )
        
        # Check AMD driver
        try:
            ps_command = '''
            Get-WmiObject Win32_VideoController | 
            Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*"} | 
            Select-Object DriverVersion | ConvertTo-Json
            '''
            
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                driver_data = json.loads(result.stdout)
                if isinstance(driver_data, dict):
                    driver_version = driver_data.get('DriverVersion', '')
                    if driver_version == self.required_driver_version:
                        env_check['amd_driver_present'] = True
                    elif driver_version:
                        env_check['recommendations'].append(
                            f"AMD driver version {driver_version} detected - may need update to {self.required_driver_version}"
                        )
                        
        except Exception as e:
            env_check['critical_issues'].append(f"Failed to check AMD driver: {e}")
        
        if not env_check['amd_driver_present']:
            env_check['critical_issues'].append(
                f"❌ AMD Adrenalin Edition {self.adrenalin_version} driver not detected"
            )
            env_check['recommendations'].append(
                "Install AMD Adrenalin Edition 23.40.27.06 for DirectML before proceeding"
            )
        
        return env_check
    
    def get_system_wide_installation_commands(self) -> Dict[str, List[str]]:
        """
        Get commands for system-wide DirectML installation
        """
        commands = {
            'preparation': [
                '# Exit any virtual environments first',
                'deactivate  # If in virtual environment',
                '',
                '# Verify system Python',
                'python --version',
                'where python'
            ],
            'directml_packages': [
                '# Install base PyTorch (CPU version first)',
                'python -m pip install --upgrade pip',
                'python -m pip install torch torchvision torchaudio',
                '',
                '# Install DirectML for PyTorch',
                'python -m pip install torch-directml',
                '',
                '# Install TensorFlow with DirectML',
                'python -m pip install tensorflow-directml',
                '',
                '# Install ONNX Runtime with DirectML',
                'python -m pip install onnxruntime-directml',
                '',
                '# Install supporting packages',
                'python -m pip install numpy pillow opencv-python transformers'
            ],
            'verification': [
                '# Test DirectML installation',
                'python -c "import torch_directml; print(f\'DirectML devices: {torch_directml.device_count()}\')"',
                'python -c "import tensorflow as tf; print(f\'TF GPU devices: {len(tf.config.list_physical_devices(\'GPU\'))}\')"',
                'python -c "import onnxruntime as ort; print(f\'ONNX providers: {ort.get_available_providers()}\')"'
            ]
        }
        
        return commands
    
    def create_installation_script(self) -> str:
        """
        Create a batch script for proper DirectML installation
        """
        commands = self.get_system_wide_installation_commands()
        
        script_content = f'''@echo off
REM AMD DirectML System-Wide Installation Script
REM For AMD Adrenalin Edition 23.40.27.06
REM IMPORTANT: Do NOT run in virtual environment!

echo ================================================
echo AMD DirectML System-Wide Installation
echo ================================================
echo.
echo IMPORTANT: This script installs DirectML packages system-wide
echo Virtual environments are NOT supported for AMD DirectML
echo Ensure AMD Adrenalin Edition 23.40.27.06 is installed first
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running as Administrator
) else (
    echo ⚠️ WARNING: Not running as Administrator
    echo Some installations may fail without admin rights
)

echo.
echo Checking Python installation...
python --version
if %errorLevel% neq 0 (
    echo ❌ Python not found in PATH
    echo Please install Python system-wide first
    pause
    exit /b 1
)

echo.
echo Checking for virtual environment...
python -c "import sys; exit(1 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 0)"
if %errorLevel% neq 0 (
    echo ❌ CRITICAL: Running in virtual environment!
    echo DirectML requires system-wide installation
    echo Please exit virtual environment and run again
    pause
    exit /b 1
)

echo ✅ Using system Python - good for DirectML
echo.

echo Installing DirectML packages...
echo ================================================

echo Step 1: Upgrading pip...
python -m pip install --upgrade pip
if %errorLevel% neq 0 (
    echo ❌ Failed to upgrade pip
    pause
    exit /b 1
)

echo Step 2: Installing PyTorch (CPU base)...
python -m pip install torch torchvision torchaudio
if %errorLevel% neq 0 (
    echo ❌ Failed to install PyTorch
    pause
    exit /b 1
)

echo Step 3: Installing torch-directml...
python -m pip install torch-directml
if %errorLevel% neq 0 (
    echo ❌ Failed to install torch-directml
    pause
    exit /b 1
)

echo Step 4: Installing tensorflow-directml...
python -m pip install tensorflow-directml
if %errorLevel% neq 0 (
    echo ❌ Failed to install tensorflow-directml
    pause
    exit /b 1
)

echo Step 5: Installing onnxruntime-directml...
python -m pip install onnxruntime-directml
if %errorLevel% neq 0 (
    echo ❌ Failed to install onnxruntime-directml
    pause
    exit /b 1
)

echo Step 6: Installing supporting packages...
python -m pip install numpy pillow opencv-python transformers accelerate diffusers
if %errorLevel% neq 0 (
    echo ⚠️ Some supporting packages failed - DirectML may still work
)

echo.
echo ================================================
echo Testing DirectML Installation
echo ================================================

echo Testing torch-directml...
python -c "import torch_directml; print(f'✅ DirectML devices: {{torch_directml.device_count()}}')" 2>nul
if %errorLevel% neq 0 (
    echo ❌ torch-directml test failed
) else (
    echo ✅ torch-directml working
)

echo Testing tensorflow-directml...
python -c "import tensorflow as tf; print(f'✅ TF GPU devices: {{len(tf.config.list_physical_devices(\"GPU\"))}}')" 2>nul
if %errorLevel% neq 0 (
    echo ❌ tensorflow-directml test failed
) else (
    echo ✅ tensorflow-directml working
)

echo Testing onnxruntime-directml...
python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print(f'✅ DirectML available: {{\"DmlExecutionProvider\" in providers}}')" 2>nul
if %errorLevel% neq 0 (
    echo ❌ onnxruntime-directml test failed
) else (
    echo ✅ onnxruntime-directml working
)

echo.
echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo To use DirectML in your projects:
echo 1. Always use system Python (not virtual environments)
echo 2. Use device = torch_directml.device() for PyTorch
echo 3. TensorFlow will automatically use DirectML GPUs
echo 4. ONNX Runtime will use DmlExecutionProvider
echo.
echo For mixed AMD/NVIDIA setups:
echo - Use system Python + DirectML for AMD GPUs
echo - Use separate virtual env + CUDA for NVIDIA GPUs
echo.
pause
'''
        
        script_path = 'install_amd_directml_system_wide.bat'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def create_device_selection_guide(self) -> str:
        """
        Create a Python guide for using DirectML with AMD GPUs
        """
        guide_content = '''"""
AMD DirectML Device Selection Guide
For use with AMD Adrenalin Edition 23.40.27.06

IMPORTANT: This code assumes DirectML packages are installed system-wide
Virtual environments may break DirectML functionality
"""

import torch
import torch_directml
import tensorflow as tf
import onnxruntime as ort
from typing import List, Optional

class AMDDirectMLDeviceManager:
    """
    Manages AMD GPU device selection for DirectML workloads
    """
    
    def __init__(self):
        self.available_devices = self.detect_directml_devices()
        
    def detect_directml_devices(self) -> List[dict]:
        """Detect available DirectML devices"""
        devices = []
        
        try:
            # PyTorch DirectML devices
            device_count = torch_directml.device_count()
            for i in range(device_count):
                devices.append({
                    'framework': 'pytorch',
                    'device_id': i,
                    'device_string': f'privateuseone:{i}',
                    'available': True
                })
        except Exception as e:
            print(f"PyTorch DirectML not available: {e}")
        
        try:
            # TensorFlow DirectML devices
            tf_devices = tf.config.list_physical_devices('GPU')
            for i, device in enumerate(tf_devices):
                devices.append({
                    'framework': 'tensorflow',
                    'device_id': i,
                    'device_string': f'/GPU:{i}',
                    'device_name': device.name,
                    'available': True
                })
        except Exception as e:
            print(f"TensorFlow DirectML not available: {e}")
        
        return devices
    
    def get_pytorch_device(self, device_id: int = 0) -> torch.device:
        """Get PyTorch DirectML device"""
        try:
            if torch_directml.device_count() > device_id:
                return torch_directml.device(device_id)
            else:
                print(f"DirectML device {device_id} not available, using CPU")
                return torch.device('cpu')
        except Exception as e:
            print(f"DirectML error: {e}, using CPU")
            return torch.device('cpu')
    
    def configure_tensorflow_directml(self):
        """Configure TensorFlow for DirectML"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Enable memory growth for DirectML
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Configured {len(gpus)} DirectML GPU(s) for TensorFlow")
                return True
            else:
                print("No DirectML GPUs found for TensorFlow")
                return False
        except Exception as e:
            print(f"TensorFlow DirectML configuration failed: {e}")
            return False
    
    def get_onnx_providers(self) -> List[str]:
        """Get ONNX Runtime providers with DirectML priority"""
        try:
            available_providers = ort.get_available_providers()
            
            # Prioritize DirectML for AMD GPUs
            preferred_order = [
                'DmlExecutionProvider',  # DirectML for AMD
                'CPUExecutionProvider'   # Fallback
            ]
            
            providers = []
            for provider in preferred_order:
                if provider in available_providers:
                    providers.append(provider)
            
            return providers
            
        except Exception as e:
            print(f"ONNX Runtime provider check failed: {e}")
            return ['CPUExecutionProvider']
    
    def test_directml_performance(self):
        """Test DirectML performance across frameworks"""
        print("🧪 Testing DirectML Performance")
        print("=" * 40)
        
        # PyTorch DirectML test
        try:
            device = self.get_pytorch_device()
            print(f"PyTorch device: {device}")
            
            import time
            start_time = time.time()
            
            # Matrix multiplication test
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            
            end_time = time.time()
            print(f"✅ PyTorch DirectML: {end_time - start_time:.3f}s for 1000x1000 matmul")
            
        except Exception as e:
            print(f"❌ PyTorch DirectML test failed: {e}")
        
        # TensorFlow DirectML test
        try:
            if self.configure_tensorflow_directml():
                start_time = time.time()
                
                with tf.device('/GPU:0'):
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                
                end_time = time.time()
                print(f"✅ TensorFlow DirectML: {end_time - start_time:.3f}s for 1000x1000 matmul")
            
        except Exception as e:
            print(f"❌ TensorFlow DirectML test failed: {e}")
        
        # ONNX Runtime test
        try:
            providers = self.get_onnx_providers()
            print(f"✅ ONNX Runtime providers: {providers}")
            
        except Exception as e:
            print(f"❌ ONNX Runtime test failed: {e}")

# Example usage for your RX 6800/6800 XT setup
if __name__ == "__main__":
    print("🚀 AMD DirectML Device Manager")
    print("For RX 6800/6800 XT with Adrenalin Edition 23.40.27.06")
    print("=" * 60)
    
    manager = AMDDirectMLDeviceManager()
    
    print(f"Available devices: {len(manager.available_devices)}")
    for device in manager.available_devices:
        print(f"  {device['framework']}: {device['device_string']}")
    
    # Test performance
    manager.test_directml_performance()
    
    print("\\n💡 Usage Tips:")
    print("- Always use system Python (not virtual environments)")
    print("- DirectML works best with latest AMD drivers")
    print("- Use device = torch_directml.device() for PyTorch")
    print("- TensorFlow automatically detects DirectML GPUs")
    print("- For mixed setups, use CUDA for NVIDIA, DirectML for AMD")
'''
        
        guide_path = 'amd_directml_device_guide.py'
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        return guide_path
    
    def run_installation_analysis(self) -> Dict[str, Any]:
        """
        Run complete installation analysis for AMD DirectML
        """
        analysis = {
            'start_time': datetime.now().isoformat(),
            'environment_check': {},
            'installation_ready': False,
            'scripts_created': [],
            'recommendations': [],
            'critical_actions': []
        }
        
        print("🔍 AMD DirectML Installation Analysis")
        print("=" * 50)
        
        # Check environment
        analysis['environment_check'] = self.check_installation_environment()
        env = analysis['environment_check']
        
        # Determine if ready for installation
        analysis['installation_ready'] = (
            not env['in_virtual_env'] and 
            env['amd_driver_present'] and 
            len(env['critical_issues']) == 0
        )
        
        if env['critical_issues']:
            print("❌ Critical Issues Found:")
            for issue in env['critical_issues']:
                print(f"   {issue}")
            analysis['critical_actions'].extend(env['critical_issues'])
        
        if env['recommendations']:
            print("💡 Recommendations:")
            for rec in env['recommendations']:
                print(f"   {rec}")
            analysis['recommendations'].extend(env['recommendations'])
        
        # Create installation scripts
        print("\\n📝 Creating installation scripts...")
        script_path = self.create_installation_script()
        guide_path = self.create_device_selection_guide()
        
        analysis['scripts_created'] = [script_path, guide_path]
        
        print(f"✅ Created: {script_path}")
        print(f"✅ Created: {guide_path}")
        
        # Final recommendations
        if analysis['installation_ready']:
            print("\\n🎉 Ready for DirectML installation!")
            print(f"Run: {script_path}")
        else:
            print("\\n⚠️ Environment needs preparation before DirectML installation")
            
            if env['in_virtual_env']:
                print("🚨 CRITICAL: Exit virtual environment first!")
                print("DirectML requires system-wide Python installation")
            
            if not env['amd_driver_present']:
                print("🚨 CRITICAL: Install AMD Adrenalin Edition 23.40.27.06 first!")
                print("Download: https://drivers.amd.com/drivers/amd-software-adrenalin-edition-23.40.27.06-win10-win11-may-rdna.exe")
        
        analysis['end_time'] = datetime.now().isoformat()
        
        # Save analysis
        with open('amd_directml_installation_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def main():
    """Run AMD DirectML installation analysis"""
    guide = AMDDirectMLInstallationGuide()
    analysis = guide.run_installation_analysis()
    
    if analysis['installation_ready']:
        print("\\n🚀 Ready to install DirectML!")
        print("Run the generated batch script to install DirectML system-wide")
    else:
        print("\\n⚠️ Please address critical issues before proceeding")

if __name__ == "__main__":
    main()
