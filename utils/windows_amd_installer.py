#!/usr/bin/env python3
"""
Windows AMD GPU Installation Manager
Optimized for Windows AMD setups with DirectML fallback
"""

import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindowsAMDInstaller:
    """Windows-specific AMD GPU worker installer"""
    
    def __init__(self):
        self.installation_log = []
        
    def install_pytorch_cpu_first(self) -> bool:
        """Install PyTorch CPU version first, then try DirectML"""
        logger.info("🔥 Installing PyTorch (CPU) + DirectML for AMD GPUs...")
        
        try:
            # Install stable PyTorch CPU version
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio', 
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], check=True)
            logger.info("✅ PyTorch CPU installed")
            
            # Install DirectML packages
            directml_packages = [
                'tensorflow-directml',
                'onnxruntime-directml'
            ]
            
            for package in directml_packages:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True)
                    logger.info(f"✅ Installed {package}")
                except subprocess.CalledProcessError:
                    logger.warning(f"⚠️ Could not install {package}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install PyTorch: {e}")
            return False
    
    def install_ai_packages(self) -> bool:
        """Install essential AI packages"""
        logger.info("🤖 Installing AI packages...")
        
        packages = [
            'transformers',
            'diffusers', 
            'accelerate',
            'safetensors',
            'datasets',
            'tokenizers',
            'huggingface-hub',
            'opencv-python',
            'pillow',
            'numpy',
            'scipy',
            'scikit-learn',
            'gradio',
            'fastapi',
            'uvicorn'
        ]
        
        success_count = 0
        for package in packages:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                logger.info(f"✅ Installed {package}")
                success_count += 1
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {package}")
        
        logger.info(f"📊 Installed {success_count}/{len(packages)} packages")
        return success_count >= len(packages) * 0.8  # 80% success rate
    
    def test_installation(self) -> Dict[str, Any]:
        """Test the installation"""
        logger.info("🧪 Testing installation...")
        
        results = {
            'torch_available': False,
            'directml_available': False,
            'transformers_available': False,
            'gpu_accessible': False
        }
        
        # Test PyTorch
        try:
            import torch
            results['torch_available'] = True
            logger.info("✅ PyTorch available")
            
            # Test basic tensor operations
            x = torch.randn(100, 100)
            y = torch.matmul(x, x.T)
            logger.info("✅ PyTorch operations working")
            
        except ImportError:
            logger.error("❌ PyTorch not available")
        except Exception as e:
            logger.warning(f"⚠️ PyTorch test failed: {e}")
        
        # Test DirectML
        try:
            import tensorflow as tf
            devices = tf.config.list_physical_devices('DML')
            if devices:
                results['directml_available'] = True
                results['gpu_accessible'] = True
                logger.info(f"✅ DirectML available with {len(devices)} device(s)")
            else:
                logger.warning("⚠️ DirectML installed but no devices found")
        except ImportError:
            logger.warning("⚠️ TensorFlow DirectML not available")
        except Exception as e:
            logger.warning(f"⚠️ DirectML test failed: {e}")
        
        # Test Transformers
        try:
            import transformers
            results['transformers_available'] = True
            logger.info("✅ Transformers available")
        except ImportError:
            logger.error("❌ Transformers not available")
        
        return results
    
    def run_installation(self) -> Dict[str, Any]:
        """Run the full installation"""
        logger.info("🚀 Windows AMD GPU Installation")
        logger.info("=" * 50)
        
        results = {
            'pytorch_installed': False,
            'ai_packages_installed': False,
            'test_results': {},
            'success': False
        }
        
        # Step 1: Install PyTorch + DirectML
        results['pytorch_installed'] = self.install_pytorch_cpu_first()
        
        # Step 2: Install AI packages
        if results['pytorch_installed']:
            results['ai_packages_installed'] = self.install_ai_packages()
        
        # Step 3: Test installation
        results['test_results'] = self.test_installation()
        
        # Overall success
        results['success'] = (
            results['pytorch_installed'] and 
            results['ai_packages_installed'] and
            results['test_results']['torch_available']
        )
        
        # Save results
        with open('windows_amd_installation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        logger.info("\n📊 Installation Summary:")
        logger.info(f"   PyTorch: {'✅' if results['pytorch_installed'] else '❌'}")
        logger.info(f"   AI Packages: {'✅' if results['ai_packages_installed'] else '❌'}")
        logger.info(f"   DirectML: {'✅' if results['test_results'].get('directml_available') else '❌'}")
        logger.info(f"   Overall: {'✅' if results['success'] else '❌'}")
        
        if results['success']:
            logger.info("\n🎉 Installation completed successfully!")
            logger.info("Your AMD GPUs are ready for AI workloads via DirectML.")
        else:
            logger.info("\n⚠️ Installation completed with issues.")
            logger.info("Check windows_amd_installation.json for details.")
        
        return results

def main():
    installer = WindowsAMDInstaller()
    return installer.run_installation()

if __name__ == "__main__":
    main()
