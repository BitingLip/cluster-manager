#!/usr/bin/env python3
"""
Cluster Task Manager for Mixed GPU Environments
Handles DirectML system-wide requirements and CUDA virtual env compatibility
"""

import torch
import sys
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MixedGPUTaskManager:
    """
    Manages task execution across mixed AMD/NVIDIA GPU environments
    """
    
    def __init__(self):
        self.environment_status = self._check_environment()
        self.available_devices = self._discover_devices()
    
    def _check_environment(self) -> Dict[str, Any]:
        """Check current environment status"""
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        return {
            'in_virtual_env': in_venv,
            'system_python': not in_venv,
            'python_executable': sys.executable
        }
    
    def _discover_devices(self) -> Dict[str, Any]:
        """Discover available devices respecting environment constraints"""
        devices = {
            'amd_directml': [],
            'nvidia_cuda': [],
            'cpu_fallback': ['cpu']
        }
        
        # AMD DirectML (only if system Python)
        if self.environment_status['system_python']:
            try:
                import torch_directml
                device_count = torch_directml.device_count()
                devices['amd_directml'] = [f'privateuseone:{i}' for i in range(device_count)]
                logger.info(f"DirectML: {device_count} AMD devices available")
            except ImportError:
                logger.warning("DirectML not available")
        else:
            logger.warning("DirectML unavailable - virtual environment detected")
        
        # NVIDIA CUDA (available in any environment)
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                devices['nvidia_cuda'] = [f'cuda:{i}' for i in range(device_count)]
                logger.info(f"CUDA: {device_count} NVIDIA devices available")
        except Exception as e:
            logger.warning(f"CUDA check failed: {e}")
        
        return devices
    
    def select_optimal_device(self, task_type: str = 'general', prefer_vendor: str = 'auto') -> Dict[str, Any]:
        """
        Select optimal device for task execution
        
        Args:
            task_type: Type of task ('inference', 'training', 'general')
            prefer_vendor: Preferred vendor ('amd', 'nvidia', 'auto')
        
        Returns:
            Device selection result with setup instructions
        """
        result = {
            'device_string': 'cpu',
            'vendor': 'CPU',
            'framework': 'torch_cpu',
            'setup_code': '',
            'warnings': [],
            'compatible': True
        }
        
        # AMD DirectML selection
        if prefer_vendor in ['amd', 'auto'] and self.available_devices['amd_directml']:
            if self.environment_status['system_python']:
                result.update({
                    'device_string': self.available_devices['amd_directml'][0],
                    'vendor': 'AMD',
                    'framework': 'torch_directml',
                    'setup_code': 'import torch_directml; device = torch_directml.device()',
                    'environment_requirement': 'system_python_only'
                })
                logger.info(f"Selected AMD DirectML device: {result['device_string']}")
                return result
            else:
                result['warnings'].append("AMD DirectML requires system Python - exit virtual environment")
        
        # NVIDIA CUDA selection
        if prefer_vendor in ['nvidia', 'auto'] and self.available_devices['nvidia_cuda']:
            result.update({
                'device_string': self.available_devices['nvidia_cuda'][0],
                'vendor': 'NVIDIA',
                'framework': 'torch_cuda',
                'setup_code': 'import torch; device = torch.device("cuda:0")',
                'environment_requirement': 'any_python_environment'
            })
            logger.info(f"Selected NVIDIA CUDA device: {result['device_string']}")
            return result
        
        # CPU fallback
        result['warnings'].append("Using CPU fallback - no compatible GPU found")
        result['setup_code'] = 'import torch; device = torch.device("cpu")'
        logger.warning("Falling back to CPU")
        
        return result
    
    def execute_task(self, task_code: str, device_preference: str = 'auto') -> Dict[str, Any]:
        """
        Execute task with optimal device selection
        
        Args:
            task_code: Python code to execute
            device_preference: Device vendor preference
        
        Returns:
            Execution result
        """
        # Select optimal device
        device_info = self.select_optimal_device(prefer_vendor=device_preference)
        
        execution_result = {
            'device_used': device_info['device_string'],
            'vendor': device_info['vendor'],
            'success': False,
            'output': '',
            'error': '',
            'warnings': device_info['warnings']
        }
        
        try:
            # Setup device
            exec(device_info['setup_code'])
            
            # Execute task code
            # Note: In a real implementation, this would be more sophisticated
            # with proper sandboxing and error handling
            local_vars = {'device': eval('device')}
            exec(task_code, globals(), local_vars)
            
            execution_result.update({
                'success': True,
                'output': f"Task executed successfully on {device_info['vendor']} device"
            })
            
        except Exception as e:
            execution_result.update({
                'success': False,
                'error': str(e)
            })
            logger.error(f"Task execution failed: {e}")
        
        return execution_result

# Usage example
if __name__ == "__main__":
    task_manager = MixedGPUTaskManager()
    
    # Example task: matrix multiplication
    task_code = '''
import torch
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
result = torch.mm(x, y)
print(f"Matrix multiplication completed on {device}")
'''
    
    # Execute with auto device selection
    result = task_manager.execute_task(task_code, device_preference='auto')
    
    print(f"Task result: {result['success']}")
    print(f"Device used: {result['device_used']} ({result['vendor']})")
    
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
