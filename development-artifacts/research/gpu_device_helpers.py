"""
Mixed GPU Device Selection Helpers
Handles DirectML system-wide requirements and CUDA virtual environment compatibility
"""

import torch
import os
import sys
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class MixedGPUDeviceSelector:
    """
    Intelligent device selector for mixed AMD/NVIDIA environments
    Respects DirectML system-wide requirements and CUDA virtual env compatibility
    """
    
    def __init__(self):
        self.environment_status = self._check_environment()
        self.available_devices = self._discover_devices()
        
    def _check_environment(self) -> Dict[str, Any]:
        """Check current Python environment status"""
        status = {
            'python_executable': sys.executable,
            'in_virtual_env': self._is_in_virtual_env(),
            'directml_compatible': False,
            'cuda_compatible': False,
            'warnings': []
        }
        
        # DirectML compatibility (requires system Python)
        status['directml_compatible'] = not status['in_virtual_env']
        
        # CUDA compatibility (works in both)
        status['cuda_compatible'] = True
        
        if status['in_virtual_env']:
            status['warnings'].append('Virtual environment detected - DirectML unavailable')
        
        return status
    
    def _is_in_virtual_env(self) -> bool:
        """Check if running in virtual environment"""
        return (hasattr(sys, 'real_prefix') or 
                (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    def _discover_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover available devices respecting environment constraints"""
        devices = {
            'amd_directml': [],
            'nvidia_cuda': [],
            'cpu_fallback': [{'device_string': 'cpu', 'name': 'CPU'}]
        }
        
        # AMD DirectML devices (only if in system Python)
        if self.environment_status['directml_compatible']:
            try:
                import torch_directml
                device_count = torch_directml.device_count()
                for i in range(device_count):
                    devices['amd_directml'].append({
                        'device_id': i,
                        'device_string': f'privateuseone:{i}',
                        'name': f'AMD DirectML Device {i}',
                        'framework': 'torch_directml',
                        'requires_system_python': True
                    })
                logger.info(f'DirectML: {device_count} AMD devices available')
            except ImportError:
                logger.warning('DirectML not installed')
            except Exception as e:
                logger.error(f'DirectML discovery failed: {e}')
        
        # NVIDIA CUDA devices (available in any environment)
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    devices['nvidia_cuda'].append({
                        'device_id': i,
                        'device_string': f'cuda:{i}',
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'framework': 'torch_cuda',
                        'virtual_env_compatible': True
                    })
                logger.info(f'CUDA: {device_count} NVIDIA devices available')
        except Exception as e:
            logger.error(f'CUDA discovery failed: {e}')
        
        return devices
    
    def get_optimal_device(self, prefer_vendor: str = 'auto', task_type: str = 'general') -> Dict[str, Any]:
        """
        Get optimal device based on environment and preferences
        
        Args:
            prefer_vendor: 'amd', 'nvidia', 'auto'
            task_type: 'training', 'inference', 'general'
        
        Returns:
            Device info dict with device, warnings, and setup instructions
        """
        result = {
            'device': None,
            'device_string': 'cpu',
            'vendor': 'CPU',
            'framework': 'torch_cpu',
            'compatible': True,
            'warnings': [],
            'setup_instructions': [],
            'environment_requirements': []
        }
        
        amd_devices = self.available_devices['amd_directml']
        nvidia_devices = self.available_devices['nvidia_cuda']
        
        # Handle AMD DirectML selection
        if prefer_vendor in ['amd', 'auto'] and amd_devices:
            if self.environment_status['directml_compatible']:
                device = amd_devices[0]
                result.update({
                    'device': device,
                    'device_string': device['device_string'],
                    'vendor': 'AMD',
                    'framework': 'torch_directml'
                })
                result['setup_instructions'] = [
                    'import torch_directml',
                    f"device = torch_directml.device({device['device_id']})",
                    'tensor = tensor.to(device)'
                ]
                result['environment_requirements'] = ['System Python required']
                logger.info(f"Selected AMD DirectML device: {device['name']}")
                return result
            else:
                result['warnings'].append('❌ AMD DirectML requires system Python - exit virtual environment')
                result['environment_requirements'] = [
                    'Exit virtual environment: deactivate',
                    'Use system Python for DirectML workloads'
                ]
        
        # Handle NVIDIA CUDA selection
        if prefer_vendor in ['nvidia', 'auto'] and nvidia_devices:
            device = nvidia_devices[0]
            result.update({
                'device': device,
                'device_string': device['device_string'],
                'vendor': 'NVIDIA',
                'framework': 'torch_cuda'
            })
            result['setup_instructions'] = [
                'import torch',
                f"device = torch.device('{device['device_string']}')",
                'tensor = tensor.to(device)'
            ]
            env_status = 'virtual env' if self.environment_status['in_virtual_env'] else 'system Python'
            result['environment_requirements'] = [f'Compatible with {env_status}']
            logger.info(f"Selected NVIDIA CUDA device: {device['name']}")
            return result
        
        # CPU fallback
        result['warnings'].append('No compatible GPU found - using CPU')
        result['setup_instructions'] = [
            'import torch',
            "device = torch.device('cpu')",
            'tensor = tensor.to(device)'
        ]
        logger.warning('Falling back to CPU')
        
        return result


def get_available_nvidia_devices() -> List[Dict[str, Any]]:
    """Get list of available NVIDIA devices"""
    selector = MixedGPUDeviceSelector()
    return selector.available_devices['nvidia_cuda']


def get_available_amd_devices() -> List[Dict[str, Any]]:
    """Get list of available AMD DirectML devices"""
    selector = MixedGPUDeviceSelector()
    return selector.available_devices['amd_directml']


def auto_select_device(prefer_vendor: str = 'auto'):
    """
    Automatically select the best available device
    
    Args:
        prefer_vendor: 'amd', 'nvidia', 'auto'
    
    Returns:
        PyTorch device object
    """
    selector = MixedGPUDeviceSelector()
    result = selector.get_optimal_device(prefer_vendor=prefer_vendor)
    
    # Print selection info
    print(f"🔧 Selected device: {result['device_string']} ({result['vendor']})")
    
    if result['warnings']:
        print("⚠️ Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result['environment_requirements']:
        print("📋 Environment requirements:")
        for req in result['environment_requirements']:
            print(f"  - {req}")
    
    # Create and return PyTorch device
    if result['framework'] == 'torch_directml':
        try:
            import torch_directml
            return torch_directml.device(result['device']['device_id'])
        except ImportError:
            return torch.device('cpu')
    elif result['framework'] == 'torch_cuda':
        return torch.device(result['device_string'])
    else:
        return torch.device('cpu')


def configure_tensorflow_gpu():
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
            # Note: tensorflow_directml import is optional
            import tensorflow_directml  # type: ignore
            logger.info("Using TensorFlow-DirectML for AMD GPUs")
            return True
        except ImportError:
            logger.warning("No GPU acceleration available for TensorFlow")
            return False
    except ImportError:
        logger.error("TensorFlow not installed")
        return False


# Legacy compatibility functions
def select_best_device(prefer_nvidia: bool = True) -> str:
    """
    Legacy function for backward compatibility
    
    Args:
        prefer_nvidia: If True, prefer NVIDIA over AMD when both available
    
    Returns:
        Device string (e.g., 'cuda:0', 'privateuseone:0', 'cpu')
    """
    prefer_vendor = 'nvidia' if prefer_nvidia else 'amd'
    selector = MixedGPUDeviceSelector()
    result = selector.get_optimal_device(prefer_vendor=prefer_vendor)
    return result['device_string']

def get_gpu_memory_info() -> Dict[str, List[Dict[str, Any]]]:
    """Get memory information for all available GPUs"""
    info = {'nvidia': [], 'amd': []}
    
    # NVIDIA memory info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['nvidia'].append({
                'device': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'available_memory_gb': torch.cuda.mem_get_info(i)[0] / (1024**3)
            })
    
    # AMD memory info (basic)
    try:
        import torch_directml  # type: ignore
        for i in range(torch_directml.device_count()):
            info['amd'].append({
                'device': i,
                'name': f'DirectML Device {i}',
                'backend': 'DirectML'
            })
    except ImportError:
        pass
    
    return info


# Usage examples
if __name__ == "__main__":
    print("🔍 Mixed GPU Device Selection Test")
    print("=" * 40)
    
    # Test device selector
    selector = MixedGPUDeviceSelector()
    
    print(f"Environment: {'Virtual Env' if selector.environment_status['in_virtual_env'] else 'System Python'}")
    print(f"DirectML Compatible: {selector.environment_status['directml_compatible']}")
    
    # Test auto selection
    print("\n🎯 Auto device selection:")
    device = auto_select_device('auto')
    print(f"Selected: {device}")
    
    # Test specific vendor preferences
    print("\n🎯 AMD preference:")
    amd_result = selector.get_optimal_device('amd')
    print(f"AMD result: {amd_result['device_string']} - {amd_result['vendor']}")
    
    print("\n🎯 NVIDIA preference:")
    nvidia_result = selector.get_optimal_device('nvidia')
    print(f"NVIDIA result: {nvidia_result['device_string']} - {nvidia_result['vendor']}")
    
    # Test legacy function
    print("\n🎯 Legacy function test:")
    best_device = select_best_device(prefer_nvidia=True)
    print(f"Legacy result: {best_device}")
    
    # Show memory info
    print("\n💾 GPU Memory Info:")
    memory_info = get_gpu_memory_info()
    print(memory_info)
    
    # Configure TensorFlow
    print("\n⚙️ TensorFlow Configuration:")
    tf_configured = configure_tensorflow_gpu()
    print(f"TensorFlow GPU configured: {tf_configured}")
