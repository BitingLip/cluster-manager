#!/usr/bin/env python3
"""
AMD DirectML Installation Guide for RX 6800/6800 XT
Proper system-wide installation for Adrenalin Edition 23.40.27.06
"""

import subprocess
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AMDDirectMLInstaller:
    """
    Handles proper DirectML installation for AMD GPUs with RX 6800/6800 XT
    """
    
    def __init__(self):
        self.system_python = sys.executable
        self.current_env_type = self.detect_environment_type()
        
    def detect_environment_type(self) -> str:
        """Detect if running in virtual environment"""
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            return 'virtual_environment'
        return 'system_python'
    
    def check_amd_driver_version(self) -> Dict[str, Any]:
        """Check AMD driver version for DirectML compatibility"""
        driver_info = {
            'compatible': False,
            'version': None,
            'directml_ready': False
        }
        
        try:
            ps_command = '''
            Get-WmiObject Win32_VideoController | 
            Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*"} | 
            Select-Object Name, DriverVersion | 
            ConvertTo-Json
            '''
            
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                
                for gpu in data:
                    driver_version = gpu.get('DriverVersion', '')
                    gpu_name = gpu.get('Name', '')
                    
                    if 'RX 6800' in gpu_name or 'Radeon' in gpu_name:
                        driver_info['version'] = driver_version
                        # Check for DirectML compatible driver
                        if driver_version == "31.0.24027.6006":
                            driver_info['compatible'] = True
                            driver_info['directml_ready'] = True
                        break
                        
        except Exception as e:
            logger.error(f"Driver check failed: {e}")
        
        return driver_info
    
    def get_correct_directml_packages(self) -> List[Dict[str, str]]:
        """
        Get the correct DirectML packages and installation methods
        """
        packages = [
            {
                'name': 'torch',
                'install_cmd': 'pip install torch torchvision torchaudio',
                'description': 'PyTorch CPU base (required for DirectML)'
            },
            {
                'name': 'torch-directml',
                'install_cmd': 'pip install torch-directml',
                'description': 'PyTorch DirectML backend for AMD GPUs'
            },
            {
                'name': 'tensorflow-cpu',
                'install_cmd': 'pip install tensorflow-cpu',
                'description': 'TensorFlow CPU (DirectML comes with specific builds)'
            },
            {
                'name': 'onnxruntime-directml',
                'install_cmd': 'pip install onnxruntime-directml',
                'description': 'ONNX Runtime with DirectML support'
            },
            {
                'name': 'numpy',
                'install_cmd': 'pip install numpy>=1.21.0',
                'description': 'Numerical computing support'
            }
        ]
        
        return packages
    
    def install_packages_system_wide(self) -> Dict[str, Any]:
        """
        Install DirectML packages system-wide (recommended approach)
        """
        installation_log = {
            'success': False,
            'packages_installed': [],
            'packages_failed': [],
            'warnings': [],
            'log_entries': []
        }
        
        if self.current_env_type == 'virtual_environment':
            installation_log['warnings'].append(
                'Running in virtual environment - DirectML may not work properly'
            )
            installation_log['warnings'].append(
                'Recommend exiting venv and using system Python'
            )
        
        packages = self.get_correct_directml_packages()
        
        # Upgrade pip first
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                installation_log['log_entries'].append('pip upgraded successfully')
            else:
                installation_log['log_entries'].append(f'pip upgrade warning: {result.stderr[:200]}')
                
        except Exception as e:
            installation_log['log_entries'].append(f'pip upgrade failed: {e}')
        
        # Install packages
        for package_info in packages:
            package_name = package_info['name']
            
            try:
                print(f"Installing {package_name}...")
                installation_log['log_entries'].append(f'Installing {package_name}...')
                
                # Use specific installation command
                if package_name == 'torch':
                    cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu']
                elif package_name == 'tensorflow-cpu':
                    cmd = [sys.executable, '-m', 'pip', 'install', 'tensorflow-cpu']
                else:
                    cmd = [sys.executable, '-m', 'pip', 'install', package_name]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    installation_log['packages_installed'].append(package_name)
                    installation_log['log_entries'].append(f'✓ {package_name} installed successfully')
                    print(f"✓ {package_name} installed successfully")
                else:
                    installation_log['packages_failed'].append({
                        'package': package_name,
                        'error': result.stderr[:500]
                    })
                    installation_log['log_entries'].append(f'✗ {package_name} failed: {result.stderr[:200]}')
                    print(f"✗ {package_name} failed")
                    
            except subprocess.TimeoutExpired:
                installation_log['packages_failed'].append({
                    'package': package_name,
                    'error': 'Installation timeout'
                })
                installation_log['log_entries'].append(f'✗ {package_name} timeout')
                print(f"✗ {package_name} timeout")
                
            except Exception as e:
                installation_log['packages_failed'].append({
                    'package': package_name,
                    'error': str(e)
                })
                installation_log['log_entries'].append(f'✗ {package_name} error: {e}')
                print(f"✗ {package_name} error: {e}")
        
        installation_log['success'] = len(installation_log['packages_failed']) == 0
        
        return installation_log
    
    def test_directml_functionality(self) -> Dict[str, Any]:
        """
        Test DirectML functionality after installation
        """
        test_results = {
            'torch_directml_devices': 0,
            'onnxruntime_directml_available': False,
            'tensorflow_available': False,
            'tests_passed': [],
            'tests_failed': [],
            'device_info': []
        }
        
        # Test torch-directml
        try:
            import torch_directml
            device_count = torch_directml.device_count()
            test_results['torch_directml_devices'] = device_count
            
            if device_count > 0:
                device = torch_directml.device()
                test_results['device_info'].append(f'torch-directml device: {device}')
                test_results['tests_passed'].append('torch-directml device detection')
                
                # Test tensor operations
                x = torch.randn(100, 100).to(device)
                y = torch.randn(100, 100).to(device)
                z = torch.mm(x, y)
                test_results['tests_passed'].append('torch-directml tensor operations')
            else:
                test_results['tests_failed'].append('torch-directml: no devices found')
                
        except ImportError:
            test_results['tests_failed'].append('torch-directml: not installed')
        except Exception as e:
            test_results['tests_failed'].append(f'torch-directml: {e}')
        
        # Test onnxruntime-directml
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'DmlExecutionProvider' in providers:
                test_results['onnxruntime_directml_available'] = True
                test_results['tests_passed'].append('onnxruntime-directml provider available')
            else:
                test_results['tests_failed'].append('onnxruntime-directml: DML provider not found')
                
        except ImportError:
            test_results['tests_failed'].append('onnxruntime: not installed')
        except Exception as e:
            test_results['tests_failed'].append(f'onnxruntime: {e}')
        
        # Test TensorFlow
        try:
            import tensorflow as tf
            test_results['tensorflow_available'] = True
            test_results['tests_passed'].append('tensorflow available')
            
            # Note: TensorFlow DirectML requires special builds
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                test_results['tests_passed'].append(f'tensorflow: {len(gpus)} GPU devices')
            else:
                test_results['tests_failed'].append('tensorflow: no GPU devices (expected with CPU build)')
                
        except ImportError:
            test_results['tests_failed'].append('tensorflow: not installed')
        except Exception as e:
            test_results['tests_failed'].append(f'tensorflow: {e}')
        
        return test_results
    
    def create_usage_example(self) -> str:
        """
        Create example script showing how to use DirectML
        """
        example_script = '''"""
DirectML Usage Example for AMD RX 6800/6800 XT
System-wide installation approach
"""

import torch
import torch_directml

def test_directml_pytorch():
    """Test PyTorch with DirectML"""
    print("Testing PyTorch DirectML...")
    
    # Check available DirectML devices
    device_count = torch_directml.device_count()
    print(f"DirectML devices available: {device_count}")
    
    if device_count == 0:
        print("No DirectML devices found!")
        return False
    
    # Use DirectML device
    device = torch_directml.device()
    print(f"Using device: {device}")
    
    # Create tensors on DirectML device
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform computation
    import time
    start_time = time.time()
    z = torch.mm(x, y)
    end_time = time.time()
    
    print(f"Matrix multiplication completed in {end_time - start_time:.3f} seconds")
    print(f"Result shape: {z.shape}")
    
    return True

def test_onnx_directml():
    """Test ONNX Runtime with DirectML"""
    print("\\nTesting ONNX Runtime DirectML...")
    
    try:
        import onnxruntime as ort
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            print("DirectML provider is available!")
            return True
        else:
            print("DirectML provider not found")
            return False
            
    except ImportError:
        print("ONNX Runtime not installed")
        return False

def main():
    """Run DirectML tests"""
    print("DirectML Test Suite for AMD RX 6800/6800 XT")
    print("=" * 50)
    
    pytorch_ok = test_directml_pytorch()
    onnx_ok = test_onnx_directml()
    
    print("\\n" + "=" * 50)
    print("Test Results:")
    print(f"PyTorch DirectML: {'✓' if pytorch_ok else '✗'}")
    print(f"ONNX DirectML: {'✓' if onnx_ok else '✗'}")
    
    if pytorch_ok or onnx_ok:
        print("\\n✓ DirectML is working!")
        print("\\nUsage tips:")
        print("- Always use system Python (not virtual environments)")
        print("- Use device = torch_directml.device() for PyTorch")
        print("- Use 'DmlExecutionProvider' for ONNX Runtime")
    else:
        print("\\n✗ DirectML setup needs attention")
        print("- Check AMD driver installation")
        print("- Ensure running with system Python")
        print("- Verify packages installed system-wide")

if __name__ == "__main__":
    main()
'''
        
        with open('directml_usage_example.py', 'w', encoding='utf-8') as f:
            f.write(example_script)
        
        return 'directml_usage_example.py'
    
    def run_full_installation(self) -> Dict[str, Any]:
        """
        Run complete DirectML installation and setup
        """
        results = {
            'start_time': datetime.now().isoformat(),
            'environment_type': self.current_env_type,
            'driver_check': {},
            'installation_results': {},
            'test_results': {},
            'example_script': '',
            'overall_success': False,
            'recommendations': []
        }
        
        print("AMD DirectML Installation for RX 6800/6800 XT")
        print("=" * 60)
        
        # Check driver
        print("Step 1: Checking AMD driver compatibility...")
        results['driver_check'] = self.check_amd_driver_version()
        
        if not results['driver_check']['directml_ready']:
            print("WARNING: AMD driver may not be DirectML compatible")
            results['recommendations'].append('Install AMD Adrenalin Edition 23.40.27.06')
        else:
            print("✓ AMD driver is DirectML compatible")
        
        # Install packages
        print("\\nStep 2: Installing DirectML packages...")
        if self.current_env_type == 'virtual_environment':
            print("WARNING: Running in virtual environment - this may cause issues")
            print("Recommend: deactivate virtual environment and use system Python")
        
        results['installation_results'] = self.install_packages_system_wide()
        
        # Test installation
        print("\\nStep 3: Testing DirectML functionality...")
        results['test_results'] = self.test_directml_functionality()
        
        # Create example
        print("\\nStep 4: Creating usage example...")
        results['example_script'] = self.create_usage_example()
        
        # Determine success
        install_success = results['installation_results']['success']
        test_success = len(results['test_results']['tests_passed']) > 0
        
        results['overall_success'] = install_success and test_success
        results['end_time'] = datetime.now().isoformat()
        
        # Final recommendations
        if not results['overall_success']:
            results['recommendations'].extend([
                'Use system Python instead of virtual environment',
                'Ensure AMD Adrenalin Edition 23.40.27.06 is installed',
                'Run as Administrator for best compatibility',
                'Restart system after driver installation'
            ])
        
        print(f"\\n{'=' * 60}")
        print(f"Installation completed: {'SUCCESS' if results['overall_success'] else 'ISSUES FOUND'}")
        print(f"Example script: {results['example_script']}")
        
        # Save results
        with open('amd_directml_installation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    """Run AMD DirectML installation"""
    installer = AMDDirectMLInstaller()
    results = installer.run_full_installation()
    
    if results['overall_success']:
        print("\\n🎉 DirectML installation successful!")
        print("🧪 Run 'python directml_usage_example.py' to test")
    else:
        print("\\n⚠️ DirectML installation had issues")
        if results['recommendations']:
            print("\\nRecommendations:")
            for rec in results['recommendations']:
                print(f"   - {rec}")

if __name__ == "__main__":
    main()
