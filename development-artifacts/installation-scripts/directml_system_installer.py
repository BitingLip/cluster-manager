#!/usr/bin/env python3
"""
DirectML System-Wide Installation Manager
Handles proper DirectML installation for AMD GPUs without virtual environment conflicts
"""

import subprocess
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DirectMLSystemInstaller:
    """
    Manages system-wide DirectML installation for AMD GPU compatibility
    """
    
    def __init__(self):
        self.system_python = sys.executable
        self.installation_log = []
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """
        Check if the system meets DirectML requirements
        """
        requirements = {
            'windows_version_ok': False,
            'amd_driver_ok': False,
            'python_version_ok': False,
            'admin_rights': False,
            'current_environment': 'unknown',
            'recommendations': []
        }
        
        # Check Windows version
        try:
            result = subprocess.run(['ver'], shell=True, capture_output=True, text=True)
            if 'Windows' in result.stdout:
                requirements['windows_version_ok'] = True
        except:
            pass
        
        # Check if running in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            requirements['current_environment'] = 'virtual_environment'
            requirements['recommendations'].append('Exit virtual environment for DirectML installation')
        else:
            requirements['current_environment'] = 'system_python'
        
        # Check Python version
        if sys.version_info >= (3, 8):
            requirements['python_version_ok'] = True
        else:
            requirements['recommendations'].append('Python 3.8+ required for DirectML')
        
        # Check admin rights (approximate)
        try:
            import ctypes
            requirements['admin_rights'] = ctypes.windll.shell32.IsUserAnAdmin()
        except:
            requirements['admin_rights'] = False
        
        if not requirements['admin_rights']:
            requirements['recommendations'].append('Run as Administrator for best compatibility')
        
        return requirements
    
    def install_directml_packages(self, use_system_python: bool = True) -> Dict[str, Any]:
        """
        Install DirectML packages using system-wide Python
        """
        installation_result = {
            'success': False,
            'packages_installed': [],
            'packages_failed': [],
            'installation_method': 'system_wide' if use_system_python else 'current_environment',
            'log': []
        }
        
        # Packages to install for DirectML support
        directml_packages = [
            'torch',
            'torchvision', 
            'torchaudio',
            'torch-directml',
            'tensorflow-directml',
            'onnxruntime-directml',
            'numpy>=1.21.0',
            'pillow>=8.0.0'
        ]
        
        python_exe = self.system_python if use_system_python else sys.executable
        
        # Upgrade pip first
        try:
            result = subprocess.run([
                python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                installation_result['log'].append('✅ pip upgraded successfully')
            else:
                installation_result['log'].append(f'⚠️ pip upgrade warning: {result.stderr}')
                
        except Exception as e:
            installation_result['log'].append(f'⚠️ pip upgrade failed: {e}')
        
        # Install each package
        for package in directml_packages:
            try:
                logger.info(f"Installing {package}...")
                installation_result['log'].append(f'Installing {package}...')
                
                result = subprocess.run([
                    python_exe, '-m', 'pip', 'install', package, '--upgrade'
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    installation_result['packages_installed'].append(package)
                    installation_result['log'].append(f'✅ {package} installed successfully')
                    logger.info(f"✅ {package} installed successfully")
                else:
                    installation_result['packages_failed'].append({
                        'package': package,
                        'error': result.stderr
                    })
                    installation_result['log'].append(f'❌ {package} failed: {result.stderr}')
                    logger.error(f"❌ {package} failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                installation_result['packages_failed'].append({
                    'package': package,
                    'error': 'Installation timeout'
                })
                installation_result['log'].append(f'❌ {package} timeout')
                logger.error(f"❌ {package} installation timeout")
                
            except Exception as e:
                installation_result['packages_failed'].append({
                    'package': package,
                    'error': str(e)
                })
                installation_result['log'].append(f'❌ {package} error: {e}')
                logger.error(f"❌ {package} error: {e}")
        
        installation_result['success'] = len(installation_result['packages_failed']) == 0
        
        return installation_result
    
    def test_directml_installation(self) -> Dict[str, Any]:
        """
        Test if DirectML installation is working correctly
        """
        test_results = {
            'torch_directml_working': False,
            'tensorflow_directml_working': False,
            'onnxruntime_directml_working': False,
            'device_count': 0,
            'test_log': []
        }
        
        # Test torch-directml
        try:
            import torch
            import torch_directml
            
            device_count = torch_directml.device_count()
            test_results['device_count'] = device_count
            test_results['torch_directml_working'] = device_count > 0
            test_results['test_log'].append(f'✅ torch-directml: {device_count} devices')
            
            # Test tensor creation
            if device_count > 0:
                device = torch_directml.device()
                test_tensor = torch.randn(10, 10).to(device)
                test_results['test_log'].append('✅ torch-directml tensor creation successful')
            
        except ImportError as e:
            test_results['test_log'].append(f'❌ torch-directml not available: {e}')
        except Exception as e:
            test_results['test_log'].append(f'❌ torch-directml error: {e}')
        
        # Test tensorflow-directml
        try:
            import tensorflow as tf
            
            # Check for DirectML devices
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                test_results['tensorflow_directml_working'] = True
                test_results['test_log'].append(f'✅ tensorflow-directml: {len(gpus)} devices')
            else:
                test_results['test_log'].append('❌ tensorflow-directml: no GPU devices found')
                
        except ImportError as e:
            test_results['test_log'].append(f'❌ tensorflow-directml not available: {e}')
        except Exception as e:
            test_results['test_log'].append(f'❌ tensorflow-directml error: {e}')
        
        # Test onnxruntime-directml
        try:
            import onnxruntime as ort
            
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                test_results['onnxruntime_directml_working'] = True
                test_results['test_log'].append('✅ onnxruntime-directml: DML provider available')
            else:
                test_results['test_log'].append('❌ onnxruntime-directml: DML provider not found')
                
        except ImportError as e:
            test_results['test_log'].append(f'❌ onnxruntime not available: {e}')
        except Exception as e:
            test_results['test_log'].append(f'❌ onnxruntime error: {e}')
        
        return test_results
    
    def create_directml_test_script(self) -> str:
        """
        Create a test script to validate DirectML installation
        """
        test_script = '''
"""
DirectML Installation Test Script
Tests torch-directml, tensorflow-directml, and onnxruntime-directml
"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("=" * 60)

# Test 1: torch-directml
print("\\n🔍 Testing torch-directml...")
try:
    import torch
    import torch_directml
    
    device_count = torch_directml.device_count()
    print(f"✅ torch-directml devices: {device_count}")
    
    if device_count > 0:
        device = torch_directml.device()
        print(f"✅ DirectML device: {device}")
        
        # Test tensor operations
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"✅ Matrix multiplication test passed: {z.shape}")
    else:
        print("❌ No DirectML devices available")
        
except ImportError as e:
    print(f"❌ torch-directml import failed: {e}")
except Exception as e:
    print(f"❌ torch-directml test failed: {e}")

# Test 2: tensorflow-directml
print("\\n🔍 Testing tensorflow-directml...")
try:
    import tensorflow as tf
    
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✅ TensorFlow GPU devices: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        
    if gpus:
        # Test simple operation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"✅ TensorFlow GPU operation test passed: {c.shape}")
    else:
        print("❌ No TensorFlow GPU devices available")
        
except ImportError as e:
    print(f"❌ tensorflow import failed: {e}")
except Exception as e:
    print(f"❌ tensorflow test failed: {e}")

# Test 3: onnxruntime-directml
print("\\n🔍 Testing onnxruntime-directml...")
try:
    import onnxruntime as ort
    
    providers = ort.get_available_providers()
    print(f"✅ ONNX Runtime providers: {providers}")
    
    if 'DmlExecutionProvider' in providers:
        print("✅ DirectML provider available")
        
        # Test session creation
        session_options = ort.SessionOptions()
        session = ort.InferenceSession(None, providers=['DmlExecutionProvider'], sess_options=session_options)
        print("✅ DirectML session creation test passed")
    else:
        print("❌ DirectML provider not available")
        
except ImportError as e:
    print(f"❌ onnxruntime import failed: {e}")
except Exception as e:
    print(f"❌ onnxruntime test failed: {e}")

print("\\n" + "=" * 60)
print("DirectML installation test completed!")
print("If you see errors, run this script with system Python (not in virtual environment)")
'''
        
        with open('test_directml_installation.py', 'w') as f:
            f.write(test_script)
        
        return 'test_directml_installation.py'
    
    def run_complete_directml_setup(self) -> Dict[str, Any]:
        """
        Run complete DirectML setup process
        """
        setup_results = {
            'start_time': datetime.now().isoformat(),
            'system_requirements': {},
            'installation_results': {},
            'test_results': {},
            'test_script_created': '',
            'success': False,
            'recommendations': []
        }
        
        print("🚀 DirectML System-Wide Installation")
        print("=" * 50)
        
        # Check system requirements
        print("🔍 Step 1: Checking system requirements...")
        setup_results['system_requirements'] = self.check_system_requirements()
        
        req = setup_results['system_requirements']
        if req['current_environment'] == 'virtual_environment':
            print("⚠️  WARNING: Running in virtual environment!")
            print("   DirectML works best with system-wide Python installation")
            setup_results['recommendations'].append('Exit virtual environment and run with system Python')
        
        # Install packages
        print("📦 Step 2: Installing DirectML packages...")
        use_system = req['current_environment'] != 'virtual_environment'
        setup_results['installation_results'] = self.install_directml_packages(use_system_python=use_system)
        
        # Test installation
        print("🧪 Step 3: Testing DirectML installation...")
        setup_results['test_results'] = self.test_directml_installation()
        
        # Create test script
        print("📝 Step 4: Creating DirectML test script...")
        setup_results['test_script_created'] = self.create_directml_test_script()
        
        # Determine overall success
        install_success = setup_results['installation_results']['success']
        test_success = any([
            setup_results['test_results']['torch_directml_working'],
            setup_results['test_results']['tensorflow_directml_working'],
            setup_results['test_results']['onnxruntime_directml_working']
        ])
        
        setup_results['success'] = install_success and test_success
        setup_results['end_time'] = datetime.now().isoformat()
        
        # Final recommendations
        if not setup_results['success']:
            setup_results['recommendations'].extend([
                'Ensure AMD Adrenalin Edition 23.40.27.06 is installed',
                'Restart system after driver installation',
                'Run installation as Administrator',
                'Use system Python (not virtual environment)',
                'Check Windows version compatibility'
            ])
        
        print(f"\n✅ DirectML setup completed: {'Success' if setup_results['success'] else 'Partial/Failed'}")
        print(f"📄 Test script created: {setup_results['test_script_created']}")
        
        # Save results
        with open('directml_installation_results.json', 'w') as f:
            json.dump(setup_results, f, indent=2)
        
        return setup_results

def main():
    """Run DirectML system-wide installation"""
    installer = DirectMLSystemInstaller()
    results = installer.run_complete_directml_setup()
    
    if results['success']:
        print("\n🎉 DirectML installation successful!")
        print("🧪 Run 'python test_directml_installation.py' to verify")
    else:
        print("\n⚠️ DirectML installation had issues.")
        print("📋 Check directml_installation_results.json for details")
        
        if results['recommendations']:
            print("\n💡 Recommendations:")
            for rec in results['recommendations']:
                print(f"   - {rec}")

if __name__ == "__main__":
    main()
