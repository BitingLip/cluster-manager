"""
Device Selection Code for mixed_gpu_workload
Generated for mixed AMD/NVIDIA environment with DirectML system-wide requirements
"""

import torch
import logging
import sys

def get_optimal_device():
    """
    Get optimal device for current environment
    Handles DirectML system-wide requirements and CUDA virtual env compatibility
    """
    device_info = {
        'device': None,
        'device_string': 'cpu',
        'vendor': 'CPU',
        'warnings': []
    }
    
    # Check environment status
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    # Try AMD DirectML (requires system Python)
    if not in_venv:  # System Python - DirectML compatible
        try:
            import torch_directml
            if torch_directml.device_count() > 0:
                device = torch_directml.device(0)
                device_info.update({
                    'device': device,
                    'device_string': 'privateuseone:0',
                    'vendor': 'AMD',
                    'framework': 'DirectML'
                })
                print("OK Using AMD DirectML (system Python)")
                return device_info
        except ImportError:
            device_info['warnings'].append("DirectML not available")
        except Exception as e:
            device_info['warnings'].append(f"DirectML error: {str(e)}")
    else:
        device_info['warnings'].append("Virtual environment detected - DirectML unavailable")
    
    # Try NVIDIA CUDA (virtual env compatible)
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            device_info.update({
                'device': device,
                'device_string': 'cuda:0',
                'vendor': 'NVIDIA',
                'framework': 'CUDA'
            })
            venv_status = "virtual env" if in_venv else "system Python"
            print(f"OK Using NVIDIA CUDA ({venv_status})")
            return device_info
    except Exception as e:
        device_info['warnings'].append(f"CUDA error: {str(e)}")
    
    # CPU fallback
    device_info['warnings'].append("Using CPU fallback")
    device_info['device'] = torch.device('cpu')
    print("WARNING Using CPU fallback - no compatible GPU found")
    
    return device_info

# Usage example for mixed_gpu_workload
if __name__ == "__main__":
    device_info = get_optimal_device()
    
    print(f"Selected device: {device_info['device_string']}")
    print(f"Vendor: {device_info['vendor']}")
    
    if device_info['warnings']:
        print("Warnings:")
        for warning in device_info['warnings']:
            print(f"  - {warning}")
    
    # Use the device
    device = device_info['device']
    
    # Example tensor operations
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.mm(x, y)
    
    print(f"OK Tensor operation completed on {device}")
