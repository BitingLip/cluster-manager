#!/usr/bin/env python3
"""
Cluster Manager Integration for DirectML System-Wide Requirements
Updates the cluster manager to properly handle mixed GPU environments
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class ClusterManagerDirectMLIntegration:
    """
    Integrates DirectML system-wide requirements into the cluster manager
    """
    
    def __init__(self):
        self.environment_status = self._check_environment()
        self.gpu_capabilities = self._assess_gpu_capabilities()
        
    def _check_environment(self) -> Dict[str, Any]:
        """Check current environment for DirectML compatibility"""
        status = {
            'python_executable': sys.executable,
            'in_virtual_env': self._is_in_venv(),
            'system_python': not self._is_in_venv(),
            'directml_ready': False,
            'cuda_ready': False
        }
        
        # DirectML readiness check
        if status['system_python']:
            try:
                import torch_directml
                status['directml_ready'] = torch_directml.device_count() > 0
            except ImportError:
                status['directml_ready'] = False
        
        # CUDA readiness check
        try:
            import torch
            status['cuda_ready'] = torch.cuda.is_available()
        except ImportError:
            status['cuda_ready'] = False
            
        return status
    
    def _is_in_venv(self) -> bool:
        """Check if running in virtual environment"""
        return (hasattr(sys, 'real_prefix') or 
                (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    def _assess_gpu_capabilities(self) -> Dict[str, Any]:
        """Assess GPU capabilities for cluster management"""
        capabilities = {
            'amd_directml': {
                'available': False,
                'device_count': 0,
                'devices': [],
                'environment_compatible': False
            },
            'nvidia_cuda': {
                'available': False,
                'device_count': 0,
                'devices': [],
                'environment_compatible': True  # CUDA works in both environments
            }
        }
        
        # AMD DirectML assessment
        if self.environment_status['directml_ready']:
            try:
                import torch_directml
                device_count = torch_directml.device_count()
                capabilities['amd_directml'].update({
                    'available': True,
                    'device_count': device_count,
                    'devices': [f'privateuseone:{i}' for i in range(device_count)],
                    'environment_compatible': self.environment_status['system_python']
                })
            except Exception as e:
                logger.warning(f"DirectML assessment failed: {e}")
        
        # NVIDIA CUDA assessment
        if self.environment_status['cuda_ready']:
            try:
                import torch
                device_count = torch.cuda.device_count()
                devices = []
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        'device_string': f'cuda:{i}',
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3)
                    })
                
                capabilities['nvidia_cuda'].update({
                    'available': True,
                    'device_count': device_count,
                    'devices': devices
                })
            except Exception as e:
                logger.warning(f"CUDA assessment failed: {e}")
        
        return capabilities
    
    def create_cluster_device_config(self) -> Dict[str, Any]:
        """
        Create cluster device configuration that respects DirectML requirements
        """
        config = {
            'cluster_id': f'mixed_gpu_cluster_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'environment_constraints': {
                'amd_directml_requires_system_python': True,
                'nvidia_cuda_supports_virtual_env': True,
                'mixed_environment_management': 'vendor_specific'
            },
            'device_pools': {},
            'task_routing': {},
            'environment_switching': {}
        }
        
        # AMD DirectML device pool
        if self.gpu_capabilities['amd_directml']['available']:
            config['device_pools']['amd_directml'] = {
                'devices': self.gpu_capabilities['amd_directml']['devices'],
                'framework': 'torch_directml',
                'environment_requirement': 'system_python_only',
                'activation_command': 'deactivate (exit any virtual env)',
                'device_selection_code': self._generate_amd_device_code(),
                'compatible_tasks': ['inference', 'training', 'image_generation']
            }
        
        # NVIDIA CUDA device pool
        if self.gpu_capabilities['nvidia_cuda']['available']:
            config['device_pools']['nvidia_cuda'] = {
                'devices': [d['device_string'] for d in self.gpu_capabilities['nvidia_cuda']['devices']],
                'framework': 'torch_cuda',
                'environment_requirement': 'system_or_virtual_env',
                'activation_command': 'activate gpu_environments/nvidia_cuda_env (optional)',
                'device_selection_code': self._generate_nvidia_device_code(),
                'compatible_tasks': ['inference', 'training', 'compute']
            }
        
        # Task routing strategy
        config['task_routing'] = {
            'default_strategy': 'vendor_preference',
            'amd_preferred_tasks': ['image_generation', 'directml_inference'],
            'nvidia_preferred_tasks': ['cuda_training', 'tensor_compute'],
            'fallback_device': 'cpu'
        }
        
        # Environment switching guide
        config['environment_switching'] = {
            'for_amd_work': [
                '1. Ensure not in virtual environment: deactivate',
                '2. Use system Python directly',
                '3. import torch_directml',
                '4. device = torch_directml.device()'
            ],
            'for_nvidia_work': [
                '1. (Optional) Activate CUDA environment',
                '2. import torch',
                '3. device = torch.device("cuda:0")'
            ]
        }
        
        return config
    
    def _generate_amd_device_code(self) -> str:
        """Generate AMD DirectML device selection code"""
        return '''
# AMD DirectML Device Selection (System Python Only)
import torch_directml

def get_amd_device():
    """Get AMD DirectML device - requires system Python"""
    if torch_directml.device_count() == 0:
        raise RuntimeError("No DirectML devices available")
    
    device = torch_directml.device(0)  # Use first AMD device
    print(f"Using AMD DirectML device: {device}")
    return device

# Usage
device = get_amd_device()
tensor = torch.randn(100, 100).to(device)
'''
    
    def _generate_nvidia_device_code(self) -> str:
        """Generate NVIDIA CUDA device selection code"""
        return '''
# NVIDIA CUDA Device Selection (Virtual Env Compatible)
import torch

def get_nvidia_device():
    """Get NVIDIA CUDA device - works in virtual environments"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    device = torch.device("cuda:0")  # Use first NVIDIA device
    print(f"Using NVIDIA CUDA device: {torch.cuda.get_device_name(0)}")
    return device

# Usage
device = get_nvidia_device()
tensor = torch.randn(100, 100).to(device)
'''
    
    def generate_cluster_task_manager(self) -> str:
        """
        Generate cluster task manager that handles mixed GPU environments
        """
        task_manager_code = '''#!/usr/bin/env python3
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
    task_code = \'''
import torch
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
result = torch.mm(x, y)
print(f"Matrix multiplication completed on {device}")
\'''
    
    # Execute with auto device selection
    result = task_manager.execute_task(task_code, device_preference='auto')
    
    print(f"Task result: {result['success']}")
    print(f"Device used: {result['device_used']} ({result['vendor']})")
    
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
'''
        
        return task_manager_code
    
    def run_cluster_integration_analysis(self) -> Dict[str, Any]:
        """
        Run complete cluster integration analysis
        """
        print("🔧 Cluster Manager DirectML Integration Analysis")
        print("=" * 55)
        
        # Environment status
        env = self.environment_status
        print(f"Environment: {'Virtual Env' if env['in_virtual_env'] else 'System Python'}")
        print(f"DirectML Ready: {'Yes' if env['directml_ready'] else 'No'}")
        print(f"CUDA Ready: {'Yes' if env['cuda_ready'] else 'No'}")
        
        # GPU capabilities
        amd_cap = self.gpu_capabilities['amd_directml']
        nvidia_cap = self.gpu_capabilities['nvidia_cuda']
        
        print(f"\\n📊 GPU Capabilities:")
        print(f"  AMD DirectML: {amd_cap['device_count']} devices ({'Compatible' if amd_cap['environment_compatible'] else 'Incompatible environment'})")
        print(f"  NVIDIA CUDA: {nvidia_cap['device_count']} devices (Always compatible)")
        
        # Generate cluster configuration
        cluster_config = self.create_cluster_device_config()
        
        # Generate task manager
        task_manager_code = self.generate_cluster_task_manager()
        
        # Save files
        config_path = 'cluster_mixed_gpu_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_config, f, indent=2, default=str)
        
        task_manager_path = 'mixed_gpu_task_manager.py'
        with open(task_manager_path, 'w', encoding='utf-8') as f:
            f.write(task_manager_code)
        
        print(f"\\n📄 Generated files:")
        print(f"  Cluster config: {config_path}")
        print(f"  Task manager: {task_manager_path}")
        
        # Analysis summary
        analysis = {
            'environment_status': env,
            'gpu_capabilities': self.gpu_capabilities,
            'cluster_config': cluster_config,
            'generated_files': [config_path, task_manager_path],
            'recommendations': self._get_integration_recommendations()
        }
        
        return analysis
    
    def _get_integration_recommendations(self) -> List[str]:
        """Get specific recommendations for cluster integration"""
        recommendations = []
        
        if self.environment_status['in_virtual_env'] and self.gpu_capabilities['amd_directml']['device_count'] > 0:
            recommendations.extend([
                "Exit virtual environment to enable DirectML integration",
                "Use system Python for cluster management with AMD GPUs"
            ])
        
        if self.gpu_capabilities['amd_directml']['available'] and self.gpu_capabilities['nvidia_cuda']['available']:
            recommendations.extend([
                "Implement vendor-specific task routing",
                "Use environment switching for optimal performance"
            ])
        
        recommendations.extend([
            "Test cluster task manager with sample workloads",
            "Monitor device utilization across vendors",
            "Implement fallback strategies for device failures"
        ])
        
        return recommendations

def main():
    """Run cluster manager DirectML integration"""
    integrator = ClusterManagerDirectMLIntegration()
    analysis = integrator.run_cluster_integration_analysis()
    
    if analysis['recommendations']:
        print("\\n💡 Integration Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\\n✅ Cluster integration analysis complete!")

if __name__ == "__main__":
    main()
