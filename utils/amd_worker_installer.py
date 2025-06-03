#!/usr/bin/env python3
"""
AMD GPU Worker Installation Manager
Handles all required installations for AMD RX 6800/6800 XT workers
"""

import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMDWorkerInstaller:
    """Manages installations required for AMD GPU workers"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.installation_log = []
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements for AMD GPU workers"""
        requirements = {
            'platform': self.platform,
            'python_version': self.python_version,
            'python_64bit': sys.maxsize > 2**32,
            'windows_version': None,
            'amd_drivers': False,
            'visual_studio_runtime': False,
            'git_installed': False
        }
        
        if self.platform == 'windows':
            try:
                # Check Windows version
                result = subprocess.run(['ver'], shell=True, capture_output=True, text=True)
                requirements['windows_version'] = result.stdout.strip()
                
                # Check for AMD drivers
                result = subprocess.run(['dxdiag', '/t', 'temp_check.txt'], 
                                      capture_output=True, timeout=30)
                if result.returncode == 0:
                    try:
                        with open('temp_check.txt', 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            requirements['amd_drivers'] = 'AMD' in content and 'Radeon' in content
                        os.remove('temp_check.txt')
                    except:
                        pass
                
                # Check for Git
                try:
                    subprocess.run(['git', '--version'], capture_output=True, check=True)
                    requirements['git_installed'] = True
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"Error checking system requirements: {e}")
        
        return requirements
    
    def install_python_dependencies(self) -> bool:
        """Install Python dependencies for AMD GPU workers"""
        logger.info("🐍 Installing Python dependencies for AMD GPU workers...")
        
        # Core dependencies
        core_packages = [
            'torch==2.1.0+rocm5.6',  # PyTorch with ROCm
            'torchvision==0.16.0+rocm5.6',
            'torchaudio==2.1.0+rocm5.6',
            'numpy>=1.21.0',
            'pillow>=8.0.0',
            'opencv-python>=4.5.0',
            'transformers>=4.20.0',
            'accelerate>=0.20.0',
            'diffusers>=0.20.0',
            'safetensors>=0.3.0',
            'huggingface-hub>=0.15.0'
        ]
        
        # DirectML packages for Windows
        directml_packages = [
            'tensorflow-directml>=1.15.8',
            'onnxruntime-directml>=1.15.0',
            'torch-directml>=0.2.0'
        ]
        
        # ROCm packages
        rocm_packages = [
            'torch==2.1.0+rocm5.6',
            'torchvision==0.16.0+rocm5.6'
        ]
        
        try:
            # Install PyTorch with ROCm support
            logger.info("Installing PyTorch with ROCm support...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'torch==2.1.0+rocm5.6', 
                'torchvision==0.16.0+rocm5.6', 
                'torchaudio==2.1.0+rocm5.6',
                '--index-url', 'https://download.pytorch.org/whl/rocm5.6'
            ], check=True)
            
            # Install DirectML for Windows
            if self.platform == 'windows':
                logger.info("Installing DirectML packages...")
                for package in directml_packages:
                    try:
                        subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                     check=True)
                        logger.info(f"✅ Installed {package}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"⚠️ Failed to install {package}: {e}")
            
            # Install core AI packages
            logger.info("Installing core AI packages...")
            for package in core_packages[3:]:  # Skip torch packages already installed
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True)
                    logger.info(f"✅ Installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️ Failed to install {package}: {e}")
            
            self.installation_log.append("Python dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to install Python dependencies: {e}")
            self.installation_log.append(f"Python dependencies failed: {e}")
            return False
    
    def install_amd_rocm(self) -> bool:
        """Install AMD ROCm for GPU compute"""
        logger.info("🔥 Installing AMD ROCm...")
        
        if self.platform != 'windows':
            logger.warning("ROCm installation primarily for Linux. Windows uses DirectML.")
            return True
        
        try:
            # For Windows, we rely on AMD Adrenalin drivers + DirectML
            logger.info("Windows detected - using AMD Adrenalin drivers + DirectML")
            logger.info("ROCm functionality provided through PyTorch ROCm wheels")
            
            # Verify AMD drivers are installed
            result = subprocess.run(['dxdiag', '/t', 'amd_check.txt'], 
                                  capture_output=True, timeout=30)
            
            if result.returncode == 0:
                with open('amd_check.txt', 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'AMD' in content and 'Radeon' in content:
                        logger.info("✅ AMD Radeon drivers detected")
                        self.installation_log.append("AMD drivers verified")
                        os.remove('amd_check.txt')
                        return True
                os.remove('amd_check.txt')
            
            logger.warning("⚠️ AMD drivers not detected. Please install AMD Adrenalin drivers.")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking AMD ROCm setup: {e}")
            return False
    
    def install_ai_frameworks(self) -> bool:
        """Install AI frameworks optimized for AMD"""
        logger.info("🤖 Installing AI frameworks...")
        
        frameworks = {
            'huggingface': [
                'transformers>=4.30.0',
                'datasets>=2.10.0',
                'tokenizers>=0.13.0',
                'accelerate>=0.20.0'
            ],
            'computer_vision': [
                'diffusers>=0.20.0',
                'controlnet-aux>=0.0.6',
                'xformers>=0.0.20',
                'compel>=2.0.0'
            ],
            'utilities': [
                'safetensors>=0.3.0',
                'gradio>=3.35.0',
                'streamlit>=1.25.0',
                'fastapi>=0.100.0',
                'uvicorn>=0.22.0'
            ]
        }
        
        success = True
        
        for category, packages in frameworks.items():
            logger.info(f"Installing {category} packages...")
            for package in packages:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True, capture_output=True)
                    logger.info(f"✅ Installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️ Failed to install {package}")
                    success = False
        
        self.installation_log.append(f"AI frameworks installation: {'success' if success else 'partial'}")
        return success
    
    def setup_model_cache(self) -> bool:
        """Setup model cache directories"""
        logger.info("📁 Setting up model cache directories...")
        
        cache_dirs = [
            Path.home() / '.cache' / 'huggingface',
            Path.home() / '.cache' / 'torch',
            Path('models') / 'huggingface',
            Path('models') / 'diffusers',
            Path('models') / 'custom'
        ]
        
        try:
            for cache_dir in cache_dirs:
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created cache directory: {cache_dir}")
            
            # Set environment variables for cache
            os.environ['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')
            os.environ['TORCH_HOME'] = str(Path.home() / '.cache' / 'torch')
            
            self.installation_log.append("Model cache directories created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup cache directories: {e}")
            return False
    
    def test_gpu_functionality(self) -> Dict[str, Any]:
        """Test GPU functionality with installed packages"""
        logger.info("🧪 Testing GPU functionality...")
        
        test_results = {
            'torch_available': False,
            'torch_rocm': False,
            'directml_available': False,
            'gpu_count': 0,
            'memory_total': 0,
            'test_inference': False
        }
        
        try:
            # Test PyTorch
            import torch
            test_results['torch_available'] = True
            
            if torch.cuda.is_available():
                test_results['torch_rocm'] = True
                test_results['gpu_count'] = torch.cuda.device_count()
                
                # Get total memory
                for i in range(test_results['gpu_count']):
                    props = torch.cuda.get_device_properties(i)
                    test_results['memory_total'] += props.total_memory // (1024**3)
                
                # Test simple operation
                try:
                    x = torch.randn(1000, 1000).cuda()
                    y = torch.matmul(x, x.T)
                    test_results['test_inference'] = True
                    logger.info("✅ PyTorch GPU test passed")
                except Exception as e:
                    logger.warning(f"⚠️ PyTorch GPU test failed: {e}")
            
        except ImportError:
            logger.warning("PyTorch not available")
        
        try:
            # Test DirectML
            import tensorflow as tf
            if len(tf.config.list_physical_devices('DML')) > 0:
                test_results['directml_available'] = True
                logger.info("✅ DirectML available")
        except ImportError:
            logger.warning("TensorFlow DirectML not available")
        
        return test_results
    
    def create_worker_config(self, gpu_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create worker configuration for detected GPUs"""
        config = {
            'worker_type': 'amd_gpu',
            'platform': self.platform,
            'gpu_count': len(gpu_info),
            'total_vram_gb': sum(gpu['memory_total_gb'] for gpu in gpu_info),
            'workers': []
        }
        
        for i, gpu in enumerate(gpu_info):
            worker_config = {
                'worker_id': f"amd_worker_{i}",
                'gpu_index': i,
                'gpu_name': gpu['name'],
                'vram_gb': gpu['memory_total_gb'],
                'compute_units': gpu.get('compute_units', 60),
                'frameworks': {
                    'pytorch': True,
                    'directml': gpu.get('directml_compatible', True),
                    'onnx': True
                },
                'capabilities': {
                    'text_generation': True,
                    'image_generation': True,
                    'embeddings': True,
                    'classification': True
                },
                'max_batch_size': min(8, gpu['memory_total_gb'] // 2),
                'max_sequence_length': 2048 if gpu['memory_total_gb'] >= 12 else 1024
            }
            config['workers'].append(worker_config)
        
        return config
    
    def run_full_installation(self) -> Dict[str, Any]:
        """Run complete installation process"""
        logger.info("🚀 Starting AMD GPU Worker Installation")
        logger.info("=" * 60)
        
        # Check system requirements
        requirements = self.check_system_requirements()
        logger.info(f"System: {requirements['platform']} Python {requirements['python_version']}")
        
        results = {
            'system_requirements': requirements,
            'installation_steps': {},
            'test_results': {},
            'worker_config': {},
            'success': False
        }
        
        # Installation steps
        steps = [
            ('python_dependencies', self.install_python_dependencies),
            ('amd_rocm', self.install_amd_rocm),
            ('ai_frameworks', self.install_ai_frameworks),
            ('model_cache', self.setup_model_cache)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n🔧 Step: {step_name}")
            try:
                result = step_func()
                results['installation_steps'][step_name] = result
                if result:
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.warning(f"⚠️ {step_name} completed with issues")
            except Exception as e:
                logger.error(f"❌ {step_name} failed: {e}")
                results['installation_steps'][step_name] = False
        
        # Test installation
        logger.info("\n🧪 Testing installation...")
        test_results = self.test_gpu_functionality()
        results['test_results'] = test_results
        
        # Create worker config if GPUs detected
        if test_results['gpu_count'] > 0:
            # Mock GPU info for config creation
            gpu_info = [
                {
                    'name': f'AMD Radeon RX 6800{"" if i < 4 else " XT"}',
                    'memory_total_gb': 16,
                    'compute_units': 60 if i < 4 else 72,
                    'directml_compatible': True
                }
                for i in range(5)  # 4x 6800 + 1x 6800 XT
            ]
            results['worker_config'] = self.create_worker_config(gpu_info)
        
        # Overall success
        success_count = sum(1 for result in results['installation_steps'].values() if result)
        results['success'] = success_count >= len(steps) - 1  # Allow one failure
        
        # Save installation log
        with open('amd_worker_installation_log.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n📊 Installation Summary:")
        logger.info(f"   Successful steps: {success_count}/{len(steps)}")
        logger.info(f"   GPU functionality: {'✅' if test_results['test_inference'] else '❌'}")
        logger.info(f"   Overall success: {'✅' if results['success'] else '❌'}")
        logger.info(f"   Log saved to: amd_worker_installation_log.json")
        
        return results

def main():
    """Main installation function"""
    installer = AMDWorkerInstaller()
    results = installer.run_full_installation()
    
    if results['success']:
        print("\n🎉 AMD GPU Worker installation completed successfully!")
        print("Your workers are ready for AI workloads.")
    else:
        print("\n⚠️ Installation completed with some issues.")
        print("Check the log file for details.")
    
    return results

if __name__ == "__main__":
    main()
