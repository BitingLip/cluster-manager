"""
Mixed GPU Device Selection Helpers
Use these functions to properly select devices in mixed AMD/NVIDIA environments
"""

import torch
import tensorflow as tf
import os
from typing import List, Optional

def get_available_nvidia_devices() -> List[int]:
    """Get list of available NVIDIA GPU indices"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def get_available_amd_devices() -> List[int]:
    """Get list of available AMD GPU indices (DirectML)"""
    try:
        import torch_directml
        return list(range(torch_directml.device_count()))
    except ImportError:
        return []

def select_best_device(prefer_nvidia: bool = True) -> str:
    """
    Select the best available device based on preference
    
    Args:
        prefer_nvidia: If True, prefer NVIDIA over AMD when both available
    
    Returns:
        Device string (e.g., 'cuda:0', 'privateuseone:0', 'cpu')
    """
    nvidia_devices = get_available_nvidia_devices()
    amd_devices = get_available_amd_devices()
    
    if prefer_nvidia and nvidia_devices:
        return f'cuda:{nvidia_devices[0]}'
    elif amd_devices:
        return f'privateuseone:{amd_devices[0]}'
    elif nvidia_devices:
        return f'cuda:{nvidia_devices[0]}'
    else:
        return 'cpu'

def configure_tensorflow_gpu():
    """Configure TensorFlow for mixed GPU environment"""
    # Try NVIDIA first
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for NVIDIA GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} NVIDIA GPU(s) for TensorFlow")
            return True
        except Exception as e:
            print(f"Failed to configure NVIDIA GPUs: {e}")
    
    # Fallback to DirectML if available
    try:
        import tensorflow_directml
        print("Using TensorFlow-DirectML for AMD GPUs")
        return True
    except ImportError:
        print("No GPU acceleration available for TensorFlow")
        return False

def get_gpu_memory_info() -> dict:
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
        import torch_directml
        for i in range(torch_directml.device_count()):
            info['amd'].append({
                'device': i,
                'name': f'DirectML Device {i}',
                'backend': 'DirectML'
            })
    except ImportError:
        pass
    
    return info

# Example usage:
if __name__ == "__main__":
    print("GPU Device Selection Helper")
    print("=" * 40)
    
    # Show available devices
    nvidia_devices = get_available_nvidia_devices()
    amd_devices = get_available_amd_devices()
    
    print(f"NVIDIA devices: {nvidia_devices}")
    print(f"AMD devices: {amd_devices}")
    
    # Select best device
    best_device = select_best_device(prefer_nvidia=True)
    print(f"Best device: {best_device}")
    
    # Configure TensorFlow
    configure_tensorflow_gpu()
    
    # Show memory info
    memory_info = get_gpu_memory_info()
    print("GPU Memory Info:")
    print(memory_info)
