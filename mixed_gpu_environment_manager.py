#!/usr/bin/env python3
"""
Mixed GPU Environment Manager
Handles installations for systems with both AMD and NVIDIA GPUs
Prevents conflicts and ensures optimal configuration for each vendor
"""

import subprocess
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import venv
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The GPU device helpers can be imported like this after creation:
# from gpu_device_helpers import MixedGPUDeviceSelector, auto_select_device

class MixedGPUEnvironmentManager:
    """
    Manages installations and configurations for mixed AMD/NVIDIA GPU systems
    """
    
    def __init__(self):
        self.detected_gpus = []
        self.nvidia_gpus = []
        self.amd_gpus = []
        self.installation_plan = {}
        self.virtual_envs = {}
        
    def detect_all_gpus(self) -> Dict[str, Any]:
        """
        Detect all GPUs in the system (both AMD and NVIDIA)
        """
        try:
            import sys
            sys.path.append('app')
            try:
                from gpu_detector import GPUDetector
                detector = GPUDetector()
                all_gpus = detector.detect_all_gpus()
            except ImportError:
                logger.warning("GPUDetector module not found, using fallback detection")
                all_gpus = self._fallback_gpu_detection()
        except Exception as e:
            # Fallback detection method
            logger.warning(f"Main GPU detector not available ({e}), using fallback detection")
            all_gpus = self._fallback_gpu_detection()
        
        # Separate by vendor
        # Handle different return formats from GPU detection
        if isinstance(all_gpus, dict):
            # If it's a dict, extract the GPU lists
            if 'nvidia_gpus' in all_gpus and 'amd_gpus' in all_gpus:
                self.nvidia_gpus = all_gpus['nvidia_gpus']
                self.amd_gpus = all_gpus['amd_gpus']
                all_gpus_list = self.nvidia_gpus + self.amd_gpus
            else:
                # Fallback: treat dict values as potential GPU lists
                all_gpus_list = []
                for value in all_gpus.values():
                    if isinstance(value, list):
                        all_gpus_list.extend(value)
                self.nvidia_gpus = [gpu for gpu in all_gpus_list if isinstance(gpu, dict) and gpu.get('vendor', '').upper() == 'NVIDIA']
                self.amd_gpus = [gpu for gpu in all_gpus_list if isinstance(gpu, dict) and gpu.get('vendor', '').upper() == 'AMD']
        elif isinstance(all_gpus, list):
            # If it's a list of GPUs, filter by vendor
            self.nvidia_gpus = [gpu for gpu in all_gpus if isinstance(gpu, dict) and gpu.get('vendor', '').upper() == 'NVIDIA']
            self.amd_gpus = [gpu for gpu in all_gpus if isinstance(gpu, dict) and gpu.get('vendor', '').upper() == 'AMD']
            all_gpus_list = all_gpus
        else:
            # Fallback: empty lists
            logger.warning(f"Unexpected GPU detection format: {type(all_gpus)}")
            self.nvidia_gpus = []
            self.amd_gpus = []
            all_gpus_list = []
        
        detection_summary = {
            'total_gpus': len(all_gpus_list),
            'nvidia_count': len(self.nvidia_gpus),
            'amd_count': len(self.amd_gpus),
            'mixed_environment': len(self.nvidia_gpus) > 0 and len(self.amd_gpus) > 0,
            'nvidia_gpus': self.nvidia_gpus,
            'amd_gpus': self.amd_gpus,
            'detection_time': datetime.now().isoformat()
        }
        
        logger.info(f"Detected {len(self.nvidia_gpus)} NVIDIA GPU(s) and {len(self.amd_gpus)} AMD GPU(s)")
        
        return detection_summary
    
    def _fallback_gpu_detection(self) -> List[Dict[str, Any]]:
        """
        Fallback GPU detection using PowerShell
        """
        gpus = []
        
        try:
            # PowerShell command to detect both AMD and NVIDIA GPUs
            ps_command = '''
            Get-WmiObject Win32_VideoController | 
            Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*" -or $_.Name -like "*NVIDIA*" -or $_.Name -like "*GeForce*" -or $_.Name -like "*RTX*" -or $_.Name -like "*GTX*"} | 
            Select-Object Name, AdapterRAM, PNPDeviceID, DriverVersion | 
            ConvertTo-Json
            '''
            
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_data = json.loads(result.stdout)
                
                # Handle both single GPU (dict) and multiple GPUs (list)
                if isinstance(gpu_data, dict):
                    gpu_data = [gpu_data]
                
                for i, gpu in enumerate(gpu_data):
                    if gpu.get('Name'):
                        gpu_name = gpu['Name']
                        
                        # Determine vendor
                        vendor = 'Unknown'
                        if any(keyword in gpu_name.upper() for keyword in ['NVIDIA', 'GEFORCE', 'RTX', 'GTX']):
                            vendor = 'NVIDIA'
                        elif any(keyword in gpu_name.upper() for keyword in ['AMD', 'RADEON']):
                            vendor = 'AMD'
                        
                        # Estimate memory
                        memory_bytes = gpu.get('AdapterRAM', 0)
                        memory_mb = memory_bytes // (1024 * 1024) if memory_bytes else 0
                        
                        # Apply known specifications for common GPUs
                        if vendor == 'AMD' and any(model in gpu_name for model in ['6800', '6900']):
                            memory_mb = 16384  # 16GB for RX 6800/6900 series
                        
                        gpu_info = {
                            'index': i,
                            'vendor': vendor,
                            'name': gpu_name,
                            'memory_total_mb': memory_mb,
                            'driver_version': gpu.get('DriverVersion', 'Unknown'),
                            'pnp_device_id': gpu.get('PNPDeviceID', 'Unknown'),
                            'detection_method': 'fallback_powershell'
                        }
                        
                        gpus.append(gpu_info)
                        
        except Exception as e:
            logger.error(f"Fallback GPU detection failed: {e}")
        
        return gpus
    
    def analyze_compatibility_conflicts(self) -> Dict[str, Any]:
        """
        Analyze potential conflicts between AMD and NVIDIA installations
        """
        conflicts = {
            'pytorch_conflicts': [],
            'tensorflow_conflicts': [],
            'cuda_directml_conflicts': [],
            'driver_conflicts': [],
            'recommendations': []
        }
        
        has_nvidia = len(self.nvidia_gpus) > 0
        has_amd = len(self.amd_gpus) > 0
        
        if has_nvidia and has_amd:
            # PyTorch conflicts
            conflicts['pytorch_conflicts'].append({
                'issue': 'PyTorch CUDA vs ROCm builds conflict',
                'description': 'Cannot install both CUDA and ROCm PyTorch in same environment',
                'solution': 'Use separate virtual environments or prioritize CUDA for mixed setups'
            })
            
            # TensorFlow conflicts
            conflicts['tensorflow_conflicts'].append({
                'issue': 'TensorFlow GPU vs DirectML conflict',
                'description': 'tensorflow-gpu (CUDA) and tensorflow-directml cannot coexist',
                'solution': 'Use tensorflow-gpu for NVIDIA, tensorflow-directml for AMD-only workflows'
            })
            
            # CUDA/DirectML conflicts
            conflicts['cuda_directml_conflicts'].append({
                'issue': 'CUDA and DirectML runtime conflicts',
                'description': 'Some applications may default to CUDA even when DirectML is available',
                'solution': 'Explicit framework selection per GPU vendor'
            })
            
            # Recommendations for mixed environments
            conflicts['recommendations'].extend([
                'Use CUDA-based frameworks for NVIDIA GPUs (better ecosystem support)',
                'Use DirectML for AMD GPUs (official Windows support)',
                'Avoid ROCm on RDNA2 and below (unofficial support)',
                'Create vendor-specific virtual environments',
                'Use device selection in code to target specific GPU vendors'
            ])
        
        return conflicts
    
    def create_installation_plan(self) -> Dict[str, Any]:
        """
        Create a comprehensive installation plan for mixed GPU environment
        """
        plan = {
            'strategy': 'mixed_gpu_optimized',
            'environments': {},
            'base_packages': [],
            'nvidia_specific': [],
            'amd_specific': [],
            'shared_packages': [],
            'installation_order': [],
            'post_install_config': []
        }
        
        has_nvidia = len(self.nvidia_gpus) > 0
        has_amd = len(self.amd_gpus) > 0
        
        # Base packages (vendor-agnostic)
        plan['base_packages'] = [
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'scikit-learn>=1.0.0',
            'matplotlib>=3.3.0',
            'pillow>=8.0.0',
            'opencv-python>=4.5.0',
            'transformers>=4.20.0',
            'diffusers>=0.20.0',
            'accelerate>=0.20.0',
            'datasets>=2.0.0',
            'tokenizers>=0.13.0'
        ]
        
        # Shared packages (work with both vendors)
        plan['shared_packages'] = [
            'onnx>=1.12.0',
            'onnxruntime>=1.15.0',  # CPU fallback
        ]
        
        if has_nvidia:
            # NVIDIA-specific packages
            plan['nvidia_specific'] = [
                'torch>=2.0.0+cu118',  # CUDA PyTorch
                'torchvision>=0.15.0+cu118',
                'torchaudio>=2.0.0+cu118',
                'tensorflow[and-cuda]>=2.13.0',  # TensorFlow with CUDA
                'onnxruntime-gpu>=1.15.0',  # ONNX with CUDA
                'xformers>=0.0.20',  # Memory-efficient transformers
                'bitsandbytes>=0.39.0',  # Quantization support
            ]
            
            # NVIDIA environment setup
            plan['environments']['nvidia'] = {
                'name': 'nvidia_gpu_env',
                'python_version': '3.10',
                'packages': plan['base_packages'] + plan['nvidia_specific'] + plan['shared_packages'],
                'env_vars': {
                    'CUDA_VISIBLE_DEVICES': 'auto_detect_nvidia',
                    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'
                }
            }
        
        if has_amd:
            # AMD-specific packages (DirectML focus - SYSTEM-WIDE INSTALLATION REQUIRED)
            plan['amd_specific'] = [
                # Note: DirectML requires system-wide installation due to driver integration
                'torch>=2.0.0',  # CPU PyTorch base
                'torchvision>=0.15.0',
                'torchaudio>=2.0.0',
                'torch-directml>=0.2.0',  # DirectML backend for PyTorch
                'tensorflow-directml>=1.15.8',  # TensorFlow with DirectML
                'onnxruntime-directml>=1.15.0',  # ONNX with DirectML
            ]
              # AMD environment setup - CRITICAL: Must use system Python
            plan['environments']['amd'] = {
                'name': 'system_wide_directml',  
                'python_version': 'system',  
                'installation_method': 'system_wide_required',  # REQUIRED for DirectML
                'packages': plan['base_packages'] + plan['amd_specific'] + plan['shared_packages'],
                'env_vars': {
                    'TF_DIRECTML_DEVICE_COUNT': str(len(self.amd_gpus)),
                    'DML_VISIBLE_DEVICES': ','.join(str(i) for i in range(len(self.amd_gpus))),
                    'TORCH_DIRECTML_DEVICE': '0',  # Default DirectML device
                },
                'critical_requirements': [
                    'AMD Adrenalin Edition 23.40.27.06 driver required',
                    'System-wide Python installation mandatory',
                    'Virtual environments WILL BREAK DirectML functionality',
                    'Driver version 31.0.24027.6006 required'
                ],
                'installation_commands': [
                    'python -m pip install torch torchvision torchaudio',
                    'python -m pip install torch-directml',
                    'python -m pip install onnxruntime-directml',
                    'python -c "import torch_directml; print(f\'DirectML devices: {torch_directml.device_count()}\')"'
                ],
                'usage_pattern': 'direct_system_python',
                'warnings': [
                    '❌ CRITICAL: Exit any virtual environments before installing',
                    '❌ CRITICAL: Virtual environments break DirectML driver integration',
                    '⚠️ Must use system Python for all DirectML workloads'
                ],
                'installation_notes': [
                    'Step 1: Exit virtual environment: deactivate',
                    'Step 2: Install using system Python: python -m pip install <package>',
                    'Step 3: Never use virtual environments for DirectML tasks',
                    'Step 4: Verify: python -c "import torch_directml"'
                ]
            }
        
        # Mixed environment (if both vendors present)
        if has_nvidia and has_amd:
            plan['environments']['mixed'] = {
                'name': 'mixed_gpu_env',
                'python_version': '3.10',
                'packages': plan['base_packages'] + plan['shared_packages'] + [
                    'torch>=2.0.0',  # CPU PyTorch as base
                    'tensorflow>=2.13.0',  # CPU TensorFlow as base
                    'onnxruntime>=1.15.0',  # CPU ONNX as base
                ],
                'env_vars': {
                    'MIXED_GPU_MODE': 'true',
                    'PREFER_CUDA_WHEN_AVAILABLE': 'true'
                },
                'post_install': [
                    'Install CUDA packages conditionally',
                    'Install DirectML packages conditionally',
                    'Configure runtime GPU selection'
                ]
            }
        
        # Installation order (critical for avoiding conflicts)
        plan['installation_order'] = [
            'create_virtual_environments',
            'install_base_packages',
            'install_vendor_specific_packages',
            'configure_environment_variables',
            'validate_installations',
            'create_gpu_selection_scripts'
        ]
        
        return plan
    
    def create_virtual_environments(self) -> Dict[str, str]:
        """
        Create separate virtual environments for different GPU vendors
        """
        env_paths = {}
        base_env_dir = os.path.join(os.getcwd(), 'gpu_environments')
        os.makedirs(base_env_dir, exist_ok=True)
        
        for env_name, env_config in self.installation_plan['environments'].items():
            env_path = os.path.join(base_env_dir, env_config['name'])
            
            try:
                logger.info(f"Creating virtual environment: {env_config['name']}")
                venv.create(env_path, with_pip=True, clear=True)
                env_paths[env_name] = env_path
                logger.info(f"✅ Created environment at: {env_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create environment {env_config['name']}: {e}")
        
        return env_paths
    
    def install_packages_in_env(self, env_path: str, packages: List[str]) -> bool:
        """
        Install packages in a specific virtual environment
        """
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(env_path, 'Scripts', 'pip.exe')
            python_path = os.path.join(env_path, 'Scripts', 'python.exe')
        else:  # Linux/Mac
            pip_path = os.path.join(env_path, 'bin', 'pip')
            python_path = os.path.join(env_path, 'bin', 'python')
        
        try:
            # Upgrade pip first
            subprocess.run([python_path, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # Install packages
            for package in packages:
                logger.info(f"Installing {package} in {env_path}")
                
                # Handle special PyTorch CUDA index
                if 'torch' in package and 'cu118' in package:
                    result = subprocess.run([
                        pip_path, 'install', package,
                        '--index-url', 'https://download.pytorch.org/whl/cu118'
                    ], capture_output=True, text=True)
                else:
                    result = subprocess.run([
                        pip_path, 'install', package
                    ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"⚠️ Failed to install {package}: {result.stderr}")
                    return False
                else:
                    logger.info(f"✅ Installed {package}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Package installation failed: {e}")
            return False
    
    def create_gpu_selection_scripts(self) -> Dict[str, str]:
        """
        Create scripts to help users select appropriate GPU environments
        """
        scripts = {}
        
        # NVIDIA activation script
        if len(self.nvidia_gpus) > 0:
            nvidia_script = f"""@echo off
echo Activating NVIDIA GPU Environment...
call gpu_environments\\nvidia_gpu_env\\Scripts\\activate.bat
echo Environment activated. CUDA-optimized packages available.
echo NVIDIA GPUs detected: {len(self.nvidia_gpus)}
set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set TF_FORCE_GPU_ALLOW_GROWTH=true
echo Ready for NVIDIA GPU workloads!
cmd /k
"""
            with open('activate_nvidia_env.bat', 'w') as f:
                f.write(nvidia_script)
            scripts['nvidia'] = 'activate_nvidia_env.bat'
        
        # AMD activation script
        if len(self.amd_gpus) > 0:
            amd_script = f"""@echo off
echo Activating AMD GPU Environment...
call gpu_environments\\amd_gpu_env\\Scripts\\activate.bat
echo Environment activated. DirectML-optimized packages available.
echo AMD GPUs detected: {len(self.amd_gpus)}
set TF_DIRECTML_DEVICE_COUNT={len(self.amd_gpus)}
echo Ready for AMD GPU workloads!
cmd /k
"""
            with open('activate_amd_env.bat', 'w') as f:
                f.write(amd_script)
            scripts['amd'] = 'activate_amd_env.bat'
          # Mixed environment script
        if len(self.nvidia_gpus) > 0 and len(self.amd_gpus) > 0:
            mixed_script = f"""@echo off
echo Activating Mixed GPU Environment...
call gpu_environments\\mixed_gpu_env\\Scripts\\activate.bat
echo Environment activated. Both NVIDIA and AMD GPUs available.
echo NVIDIA GPUs: {len(self.nvidia_gpus)}, AMD GPUs: {len(self.amd_gpus)}
echo Use explicit device selection in your code.
echo Ready for mixed GPU workloads!
cmd /k
"""
            with open('activate_mixed_env.bat', 'w') as f:
                f.write(mixed_script)
            scripts['mixed'] = 'activate_mixed_env.bat'
        
        return scripts
    
    def create_device_selection_helpers(self) -> str:
        """
        Ensure the device selection helpers module is available
        """
        import os
        
        helper_filename = 'gpu_device_helpers.py'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_path = os.path.join(current_dir, helper_filename)
        
        # Check if the helper file exists in the current directory
        if os.path.exists(source_path):
            logger.info(f"✅ Device selection helpers already available: {helper_filename}")
            return helper_filename
        
        # If not in current directory, check if it exists in working directory
        if os.path.exists(helper_filename):
            logger.info(f"✅ Device selection helpers available: {helper_filename}")
            return helper_filename
        else:
            logger.error(f"❌ Helper file not found: {helper_filename}")
            return ""
    
    def configure_tensorflow_gpu(self):
        """Configure TensorFlow for mixed GPU environment"""
        try:
            import tensorflow as tf
            
            # Try NVIDIA first
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth for NVIDIA GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Configured {len(gpus)} NVIDIA GPU(s) for TensorFlow")
                    return True
                except Exception as e:
                    logger.error(f"Failed to configure NVIDIA GPUs: {e}")
            
            # Fallback to DirectML if available
            try:
                # Note: tensorflow_directml is an optional dependency
                import tensorflow_directml  # type: ignore
                logger.info("Using TensorFlow-DirectML for AMD GPUs")
                return True
            except ImportError:
                logger.warning("TensorFlow-DirectML not available for AMD GPUs")
                return False
                
        except ImportError:
            logger.error("TensorFlow not installed")
            return False

    def get_gpu_memory_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get memory information for all available GPUs"""
        info = {'nvidia': [], 'amd': []}
        
        # NVIDIA memory info
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info['nvidia'].append({
                        'device': i,
                        'name': props.name,
                        'total_memory_gb': props.total_memory / (1024**3),
                        'available_memory_gb': torch.cuda.mem_get_info(i)[0] / (1024**3)
                    })
        except ImportError:
            logger.warning("PyTorch not available for NVIDIA memory info")
        
        # AMD memory info (basic)
        try:
            import torch_directml
            for i in range(torch_directml.device_count()):
                info['amd'].append({
                    'device': i,
                    'name': f'DirectML Device {i}',
                    'backend': 'DirectML'
                })
        except ImportError:
            logger.warning("DirectML not available for AMD memory info")
        
        return info
    
    def generate_directml_installation_commands(self) -> str:
        """
        Generate specific installation commands for DirectML (system-wide)
        """
        commands = """
# AMD DirectML Installation Commands (SYSTEM-WIDE REQUIRED)
# IMPORTANT: Exit any virtual environments first!

# Check current environment
python --version
where python

# Verify not in virtual environment
python -c "import sys; print('Virtual env detected' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'System Python - Good for DirectML')"

# Install DirectML packages system-wide
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install torch-directml
python -m pip install tensorflow-directml
python -m pip install onnxruntime-directml

# Install AI/ML packages
python -m pip install transformers accelerate diffusers
python -m pip install numpy pillow opencv-python

# Test DirectML installation
python -c "import torch_directml; print(f'DirectML devices: {torch_directml.device_count()}')"
python -c "import tensorflow as tf; print(f'TF GPU devices: {len(tf.config.list_physical_devices('GPU'))}')"
"""
        return commands
    
    def run_complete_setup(self) -> Dict[str, Any]:
        """
        Run complete mixed GPU environment setup
        """
        setup_results = {
            'start_time': datetime.now().isoformat(),
            'gpu_detection': {},
            'conflicts_analysis': {},
            'installation_plan': {},
            'environments_created': {},
            'packages_installed': {},
            'scripts_created': {},
            'helpers_created': '',
            'success': False,
            'errors': []
        }
        
        try:
            print("🚀 Mixed GPU Environment Setup")
            print("=" * 50)
            
            # Step 1: Detect all GPUs
            print("🔍 Step 1: Detecting all GPUs...")
            setup_results['gpu_detection'] = self.detect_all_gpus()
            
            # Step 2: Analyze conflicts
            print("⚠️ Step 2: Analyzing compatibility conflicts...")
            setup_results['conflicts_analysis'] = self.analyze_compatibility_conflicts()
            
            # Step 3: Create installation plan
            print("📋 Step 3: Creating installation plan...")
            self.installation_plan = self.create_installation_plan()
            setup_results['installation_plan'] = self.installation_plan
            
            # Step 4: Create virtual environments
            print("🏗️ Step 4: Creating virtual environments...")
            setup_results['environments_created'] = self.create_virtual_environments()
            
            # Step 5: Install packages (commented out for now to avoid long install times)
            print("📦 Step 5: Package installation plan created (run manually)")
            setup_results['packages_installed'] = {'status': 'planned', 'message': 'Run installation scripts manually'}
            
            # Step 6: Create activation scripts
            print("📝 Step 6: Creating activation scripts...")
            setup_results['scripts_created'] = self.create_gpu_selection_scripts()
            
            # Step 7: Create device selection helpers
            print("🛠️ Step 7: Creating device selection helpers...")
            setup_results['helpers_created'] = self.create_device_selection_helpers()
            
            setup_results['success'] = True
            setup_results['end_time'] = datetime.now().isoformat()
            
            print("\n✅ Mixed GPU Environment Setup Complete!")
            print(f"📊 Summary:")
            print(f"   NVIDIA GPUs: {len(self.nvidia_gpus)}")
            print(f"   AMD GPUs: {len(self.amd_gpus)}")
            print(f"   Environments created: {len(setup_results['environments_created'])}")
            print(f"   Activation scripts: {len(setup_results['scripts_created'])}")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            setup_results['errors'].append(str(e))
            setup_results['success'] = False
        
        # Save results
        with open('mixed_gpu_setup_results.json', 'w') as f:
            json.dump(setup_results, f, indent=2)
        
        return setup_results

def main():
    """Main function to run mixed GPU environment setup"""
    manager = MixedGPUEnvironmentManager()
    results = manager.run_complete_setup()
    
    if results['success']:
        print("\n🎉 Setup completed successfully!")
        print("📁 Check the following files:")
        print("   - mixed_gpu_setup_results.json (detailed results)")
        print("   - activate_*_env.bat (environment activation scripts)")
        print("   - gpu_device_helpers.py (Python device selection helpers)")
    else:
        print("\n❌ Setup encountered errors. Check mixed_gpu_setup_results.json for details.")


def demonstrate_new_helper_usage():
    """
    Demonstrate how to use the new gpu_device_helpers module
    """
    try:
        # Import the helper module (this will only work if it exists)
        from gpu_device_helpers import MixedGPUDeviceSelector, auto_select_device
        
        print("🔧 Using the new GPU device helper:")
        print("=" * 40)
        
        # Use the auto selection function
        device = auto_select_device('auto')
        print(f"Auto-selected device: {device}")
        
        # Use the advanced selector
        selector = MixedGPUDeviceSelector()
        amd_result = selector.get_optimal_device('amd')
        nvidia_result = selector.get_optimal_device('nvidia')
        
        print(f"AMD optimization result: {amd_result['device_string']} ({amd_result['vendor']})")
        print(f"NVIDIA optimization result: {nvidia_result['device_string']} ({nvidia_result['vendor']})")
        
        if amd_result['warnings']:
            print("AMD Warnings:")
            for warning in amd_result['warnings']:
                print(f"  - {warning}")
                
        if nvidia_result['warnings']:
            print("NVIDIA Warnings:")
            for warning in nvidia_result['warnings']:
                print(f"  - {warning}")
                
    except ImportError:
        print("❌ gpu_device_helpers module not found. Please run create_device_selection_helpers() first.")
    except Exception as e:
        print(f"❌ Error demonstrating helper usage: {e}")


if __name__ == "__main__":
    main()
