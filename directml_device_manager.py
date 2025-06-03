#!/usr/bin/env python3
"""
DirectML-Aware Device Selection Manager
Handles proper device selection for mixed AMD/NVIDIA GPU setups with DirectML system-wide requirements
"""

import torch
import logging
import os
import sys
import json
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DirectMLDeviceManager:
    """
    Manages device selection for mixed GPU environments with DirectML system-wide requirements
    """
    
    def __init__(self):
        self.directml_available = self._check_directml_availability()
        self.cuda_available = self._check_cuda_availability()
        self.available_devices = self._discover_available_devices()
        self.environment_status = self._check_environment_status()
        
    def _check_directml_availability(self) -> bool:
        """Check if DirectML is properly installed and accessible"""
        try:
            import torch_directml
            device_count = torch_directml.device_count()
            logger.info(f"DirectML available with {device_count} devices")
            return device_count > 0
        except ImportError:
            logger.warning("DirectML not available - torch_directml not installed")
            return False
        except Exception as e:
            logger.error(f"DirectML check failed: {e}")
            return False
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is properly installed and accessible"""
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                logger.info(f"CUDA available with {device_count} devices")
                return True
            else:
                logger.info("CUDA not available")
                return False
        except Exception as e:
            logger.error(f"CUDA check failed: {e}")
            return False
    
    def _check_environment_status(self) -> Dict[str, Any]:
        """Check current Python environment status for DirectML compatibility"""
        status = {
            'python_executable': sys.executable,
            'in_virtual_env': False,
            'system_python': False,
            'directml_compatible': False,
            'warnings': []
        }
        
        # Check if in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            status['in_virtual_env'] = True
            status['warnings'].append("Running in virtual environment - DirectML may not work properly")
        else:
            status['system_python'] = True
        
        # DirectML compatibility check
        status['directml_compatible'] = (
            status['system_python'] and 
            self.directml_available and 
            not status['in_virtual_env']
        )
        
        if not status['directml_compatible'] and self.directml_available:
            status['warnings'].append("DirectML detected but environment not compatible")
        
        return status
    
    def _discover_available_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all available GPU devices"""
        devices = {
            'amd_directml': [],
            'nvidia_cuda': [],
            'fallback': []
        }
        
        # AMD DirectML devices
        if self.directml_available:
            try:
                import torch_directml
                device_count = torch_directml.device_count()
                for i in range(device_count):
                    devices['amd_directml'].append({
                        'device_id': i,
                        'device_string': f'privateuseone:{i}',
                        'framework': 'torch_directml',
                        'vendor': 'AMD',
                        'available': True,
                        'requires_system_python': True
                    })
            except Exception as e:
                logger.error(f"Failed to enumerate DirectML devices: {e}")
        
        # NVIDIA CUDA devices
        if self.cuda_available:
            try:
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    device_props = torch.cuda.get_device_properties(i)
                    devices['nvidia_cuda'].append({
                        'device_id': i,
                        'device_string': f'cuda:{i}',
                        'framework': 'torch_cuda',
                        'vendor': 'NVIDIA',
                        'name': device_props.name,
                        'memory_gb': device_props.total_memory / (1024**3),
                        'available': True,
                        'virtual_env_compatible': True
                    })
            except Exception as e:
                logger.error(f"Failed to enumerate CUDA devices: {e}")
        
        # CPU fallback
        devices['fallback'].append({
            'device_id': 0,
            'device_string': 'cpu',
            'framework': 'torch_cpu',
            'vendor': 'CPU',
            'available': True,
            'universal_compatible': True
        })
        
        return devices
    
    def get_recommended_device(self, task_type: str = 'general', prefer_vendor: str = 'auto') -> Dict[str, Any]:
        """
        Get recommended device based on task type and environment
        
        Args:
            task_type: 'general', 'training', 'inference', 'mixed'
            prefer_vendor: 'amd', 'nvidia', 'auto'
        
        Returns:
            Device recommendation with compatibility info
        """
        recommendation = {
            'device': None,
            'device_string': 'cpu',
            'vendor': 'CPU',
            'framework': 'torch_cpu',
            'compatible': True,
            'warnings': [],
            'setup_instructions': []
        }
        
        # Check environment compatibility for DirectML
        if prefer_vendor in ['amd', 'auto'] and self.directml_available:
            if not self.environment_status['directml_compatible']:
                if self.environment_status['in_virtual_env']:
                    recommendation['warnings'].append(
                        "❌ CRITICAL: DirectML requires system Python - exit virtual environment"
                    )
                    recommendation['setup_instructions'].append(
                        "1. Exit virtual environment: deactivate"
                    )
                    recommendation['setup_instructions'].append(
                        "2. Use system Python for DirectML workloads"
                    )
                    recommendation['compatible'] = False
                else:
                    recommendation['warnings'].append(
                        "DirectML available but environment check failed"
                    )
        
        # Device selection logic
        if self.environment_status['directml_compatible'] and (prefer_vendor in ['amd', 'auto']):
            # AMD DirectML (system Python only)
            amd_devices = self.available_devices['amd_directml']
            if amd_devices:
                device = amd_devices[0]  # Use first AMD device
                recommendation.update({
                    'device': device,
                    'device_string': device['device_string'],
                    'vendor': 'AMD',
                    'framework': 'torch_directml',
                    'compatible': True
                })
                recommendation['setup_instructions'].extend([
                    "Using AMD DirectML (system Python)",
                    "import torch_directml",
                    f"device = torch_directml.device({device['device_id']})"
                ])
        
        elif self.cuda_available and (prefer_vendor in ['nvidia', 'auto']):
            # NVIDIA CUDA (virtual env compatible)
            nvidia_devices = self.available_devices['nvidia_cuda']
            if nvidia_devices:
                device = nvidia_devices[0]  # Use first NVIDIA device
                recommendation.update({
                    'device': device,
                    'device_string': device['device_string'],
                    'vendor': 'NVIDIA',
                    'framework': 'torch_cuda',
                    'compatible': True
                })
                recommendation['setup_instructions'].extend([
                    "Using NVIDIA CUDA (virtual env OK)",
                    "import torch",
                    f"device = torch.device('{device['device_string']}')"
                ])
        
        else:
            # CPU fallback
            recommendation['warnings'].append("Using CPU fallback - no compatible GPU found")
            recommendation['setup_instructions'].extend([
                "Using CPU fallback",
                "import torch",
                "device = torch.device('cpu')"
            ])
        
        return recommendation
    
    def create_device_selection_code(self, task_name: str = "gpu_task") -> str:
        """
        Generate Python code for proper device selection
        """
        code_template = f'''"""
Device Selection Code for {task_name}
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
    device_info = {{
        'device': None,
        'device_string': 'cpu',
        'vendor': 'CPU',
        'warnings': []
    }}
    
    # Check environment status
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    # Try AMD DirectML (requires system Python)
    if not in_venv:  # System Python - DirectML compatible
        try:
            import torch_directml
            if torch_directml.device_count() > 0:
                device = torch_directml.device(0)
                device_info.update({{
                    'device': device,
                    'device_string': 'privateuseone:0',
                    'vendor': 'AMD',
                    'framework': 'DirectML'
                }})                print("OK Using AMD DirectML (system Python)")
                return device_info
        except ImportError:
            device_info['warnings'].append("DirectML not available")
        except Exception as e:
            device_info['warnings'].append(f"DirectML error: {e}")
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
        device_info['warnings'].append(f"CUDA error: {e}")
    
    # CPU fallback
    device_info['warnings'].append("Using CPU fallback")
    device_info['device'] = torch.device('cpu')
    print("WARNING Using CPU fallback - no compatible GPU found")
    
    return device_info

# Usage example for {task_name}
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
'''
        
        return code_template
    
    def store_environment_insights(self):
        """Store key insights about the current environment to memory"""
        insights = {
            'directml_status': self.directml_available,
            'cuda_status': self.cuda_available,
            'environment_status': self.environment_status,
            'device_count': {
                'amd': len(self.available_devices['amd_directml']),
                'nvidia': len(self.available_devices['nvidia_cuda'])
            },
            'compatibility_summary': {
                'directml_requires_system_python': True,
                'cuda_supports_virtual_env': True,
                'mixed_setup_requires_careful_management': True
            }
        }
        
        return insights
    
    def run_device_management_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive device management analysis
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'environment_status': self.environment_status,
            'available_devices': self.available_devices,
            'compatibility_check': {},
            'recommendations': [],
            'generated_files': []
        }
        
        print("🔍 DirectML-Aware Device Management Analysis")
        print("=" * 55)
        
        # Environment status
        env = self.environment_status
        print(f"Python Environment: {env['python_executable']}")
        print(f"Virtual Environment: {'Yes' if env['in_virtual_env'] else 'No'}")
        print(f"DirectML Compatible: {'Yes' if env['directml_compatible'] else 'No'}")
        
        if env['warnings']:
            print("\\n⚠️ Environment Warnings:")
            for warning in env['warnings']:
                print(f"  - {warning}")
        
        # Device summary
        amd_count = len(self.available_devices['amd_directml'])
        nvidia_count = len(self.available_devices['nvidia_cuda'])
        
        print(f"\\n📊 Available Devices:")
        print(f"  AMD DirectML: {amd_count} devices")
        print(f"  NVIDIA CUDA: {nvidia_count} devices")
        
        # Compatibility analysis
        analysis['compatibility_check'] = {
            'amd_directml_ready': env['directml_compatible'],
            'nvidia_cuda_ready': self.cuda_available,
            'mixed_setup_viable': amd_count > 0 and nvidia_count > 0,
            'environment_issues': len(env['warnings']) > 0
        }
        
        # Generate recommendations
        if env['in_virtual_env'] and amd_count > 0:
            analysis['recommendations'].append(
                "Exit virtual environment to use AMD DirectML"
            )
        
        if amd_count > 0 and nvidia_count > 0:
            analysis['recommendations'].append(
                "Mixed setup detected - use environment-specific device selection"
            )
        
        if not env['directml_compatible'] and amd_count > 0:
            analysis['recommendations'].append(
                "Install DirectML packages using system Python"
            )
        
        # Generate device selection code
        device_code_path = 'gpu_device_selector.py'
        device_code = self.create_device_selection_code("mixed_gpu_workload")
          with open(device_code_path, 'w', encoding='utf-8') as f:
            f.write(device_code)
        
        analysis['generated_files'].append(device_code_path)
        print(f"\\n📄 Generated: {device_code_path}")
          # Save analysis
        analysis_path = 'device_management_analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        analysis['generated_files'].append(analysis_path)
        print(f"📄 Analysis saved: {analysis_path}")
        
        return analysis

def main():
    """Run DirectML-aware device management analysis"""
    manager = DirectMLDeviceManager()
    analysis = manager.run_device_management_analysis()
    
    # Display final recommendations
    if analysis['recommendations']:
        print("\\n💡 Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Usage guidance
    env_status = analysis['environment_status']
    if env_status['directml_compatible']:
        print("\\n🎉 DirectML ready - use system Python for AMD GPU workloads")
    elif env_status['in_virtual_env']:
        print("\\n⚠️ Exit virtual environment to enable DirectML")
    else:
        print("\\n💡 Install DirectML packages using system Python")

if __name__ == "__main__":
    main()
