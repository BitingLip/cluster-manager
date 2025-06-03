#!/usr/bin/env python3
"""
GPU Compatibility and Conflict Resolution Manager
Handles mixed AMD/NVIDIA environments and prevents installation conflicts
"""

import json
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUVendorSupport:
    """Define support levels for different GPU vendors"""
    nvidia_cuda: bool = False
    nvidia_tensorrt: bool = False
    amd_rocm: bool = False
    amd_directml: bool = False
    intel_oneapi: bool = False

@dataclass
class FrameworkCompatibility:
    """Framework compatibility with different GPU vendors"""
    name: str
    version: str
    nvidia_support: GPUVendorSupport
    amd_support: GPUVendorSupport
    conflicts_with: List[str]
    installation_notes: str

class GPUCompatibilityManager:
    """
    Manages compatibility between different GPU vendors and frameworks
    """
    
    def __init__(self):
        self.framework_matrix = self._build_compatibility_matrix()
        
    def _build_compatibility_matrix(self) -> Dict[str, FrameworkCompatibility]:
        """
        Build comprehensive compatibility matrix for GPU frameworks
        """
        matrix = {}
        
        # PyTorch variants
        matrix['pytorch_cuda'] = FrameworkCompatibility(
            name="PyTorch with CUDA",
            version=">=2.0.0+cu118",
            nvidia_support=GPUVendorSupport(nvidia_cuda=True, nvidia_tensorrt=True),
            amd_support=GPUVendorSupport(),
            conflicts_with=['pytorch_rocm', 'pytorch_directml'],
            installation_notes="Install from CUDA index: https://download.pytorch.org/whl/cu118"
        )
        
        matrix['pytorch_rocm'] = FrameworkCompatibility(
            name="PyTorch with ROCm",
            version=">=2.0.0+rocm5.6",
            nvidia_support=GPUVendorSupport(),
            amd_support=GPUVendorSupport(amd_rocm=True),
            conflicts_with=['pytorch_cuda', 'pytorch_directml'],
            installation_notes="NOT recommended for RDNA2 (RX 6000) - use DirectML instead"
        )
        
        matrix['pytorch_directml'] = FrameworkCompatibility(
            name="PyTorch with DirectML",
            version="torch-directml>=0.2.0",
            nvidia_support=GPUVendorSupport(),
            amd_support=GPUVendorSupport(amd_directml=True),
            conflicts_with=['pytorch_cuda', 'pytorch_rocm'],
            installation_notes="Recommended for AMD RDNA2+ on Windows"
        )
        
        # TensorFlow variants
        matrix['tensorflow_gpu'] = FrameworkCompatibility(
            name="TensorFlow with CUDA",
            version="tensorflow[and-cuda]>=2.13.0",
            nvidia_support=GPUVendorSupport(nvidia_cuda=True),
            amd_support=GPUVendorSupport(),
            conflicts_with=['tensorflow_directml'],
            installation_notes="Official NVIDIA GPU support"
        )
        
        matrix['tensorflow_directml'] = FrameworkCompatibility(
            name="TensorFlow with DirectML",
            version="tensorflow-directml>=1.15.8",
            nvidia_support=GPUVendorSupport(),
            amd_support=GPUVendorSupport(amd_directml=True),
            conflicts_with=['tensorflow_gpu'],
            installation_notes="Official AMD GPU support for Windows"
        )
        
        # ONNX Runtime variants
        matrix['onnxruntime_gpu'] = FrameworkCompatibility(
            name="ONNX Runtime with CUDA",
            version="onnxruntime-gpu>=1.15.0",
            nvidia_support=GPUVendorSupport(nvidia_cuda=True),
            amd_support=GPUVendorSupport(),
            conflicts_with=['onnxruntime_directml'],
            installation_notes="NVIDIA GPU acceleration"
        )
        
        matrix['onnxruntime_directml'] = FrameworkCompatibility(
            name="ONNX Runtime with DirectML",
            version="onnxruntime-directml>=1.15.0",
            nvidia_support=GPUVendorSupport(),
            amd_support=GPUVendorSupport(amd_directml=True),
            conflicts_with=['onnxruntime_gpu'],
            installation_notes="AMD GPU acceleration via DirectML"
        )
        
        return matrix
    
    def detect_gpu_architecture_support(self, gpu_info: Dict[str, Any]) -> Dict[str, bool]:
        """
        Detect what frameworks are supported by specific GPU architecture
        """
        vendor = gpu_info.get('vendor', '').upper()
        gpu_name = gpu_info.get('name', '').lower()
        
        support = {
            'cuda_support': False,
            'rocm_support': False,
            'directml_support': False,
            'rocm_recommended': False,
            'directml_recommended': False
        }
        
        if vendor == 'NVIDIA':
            support['cuda_support'] = True
            support['directml_support'] = True  # DirectML works on NVIDIA too
            
        elif vendor == 'AMD':
            support['directml_support'] = True
            
            # ROCm support check (architecture dependent)
            if any(arch in gpu_name for arch in ['vega', 'navi 10', 'navi 20']):
                support['rocm_support'] = True
                
            # Check if RDNA2 or below (NOT recommended for ROCm)
            if any(model in gpu_name for model in ['6800', '6900', '6700', '6600', '6500', '6400']):
                support['rocm_support'] = False  # Technically possible but not recommended
                support['directml_recommended'] = True
                
            # RDNA3 has better ROCm support
            elif any(model in gpu_name for model in ['7800', '7900', '7700', '7600']):
                support['rocm_support'] = True
                support['rocm_recommended'] = True
                support['directml_recommended'] = True
            
            # Older architectures
            elif any(arch in gpu_name for arch in ['polaris', 'vega']):
                support['rocm_support'] = True
                support['rocm_recommended'] = True
        
        return support
    
    def analyze_mixed_environment_conflicts(self, nvidia_gpus: List[Dict], amd_gpus: List[Dict]) -> Dict[str, Any]:
        """
        Analyze conflicts in mixed AMD/NVIDIA environment
        """
        analysis = {
            'environment_type': 'mixed_vendor',
            'total_gpus': len(nvidia_gpus) + len(amd_gpus),
            'nvidia_count': len(nvidia_gpus),
            'amd_count': len(amd_gpus),
            'conflicts': [],
            'recommendations': [],
            'installation_strategies': []
        }
        
        # Framework conflicts
        conflicts = [
            {
                'category': 'PyTorch',
                'issue': 'Cannot install both CUDA and ROCm PyTorch builds',
                'severity': 'HIGH',
                'impact': 'Installation will fail or one will override the other',
                'solutions': [
                    'Use separate virtual environments for each vendor',
                    'Prioritize CUDA PyTorch + DirectML for mixed setups',
                    'Use CPU PyTorch + manual device selection'
                ]
            },
            {
                'category': 'TensorFlow',
                'issue': 'tensorflow-gpu and tensorflow-directml conflict',
                'severity': 'HIGH',
                'impact': 'Cannot install both in same environment',
                'solutions': [
                    'Use tensorflow-gpu for NVIDIA workflows',
                    'Use tensorflow-directml for AMD workflows',
                    'Create vendor-specific environments'
                ]
            },
            {
                'category': 'ONNX Runtime',
                'issue': 'onnxruntime-gpu and onnxruntime-directml conflict',
                'severity': 'MEDIUM',
                'impact': 'GPU acceleration limited to one vendor',
                'solutions': [
                    'Install vendor-specific ONNX runtime per environment',
                    'Use CPU version as fallback'
                ]
            },
            {
                'category': 'Memory Management',
                'issue': 'CUDA and DirectML memory allocation conflicts',
                'severity': 'MEDIUM',
                'impact': 'Potential memory leaks or allocation failures',
                'solutions': [
                    'Explicit device cleanup between vendor switches',
                    'Process isolation for different vendors'
                ]
            }
        ]
        
        analysis['conflicts'] = conflicts
        
        # Recommendations for mixed environment
        recommendations = [
            'Strategy 1: Vendor-Specific Environments (Recommended)',
            '  - Create separate virtual environments for NVIDIA and AMD',
            '  - Install vendor-optimized packages in each environment',
            '  - Use activation scripts to switch between vendors',
            '',
            'Strategy 2: CUDA-Primary with DirectML Fallback',
            '  - Install CUDA versions of frameworks as primary',
            '  - Use DirectML for AMD-specific workloads',
            '  - Requires careful device selection in code',
            '',
            'Strategy 3: Framework-Specific Separation',
            '  - PyTorch environment with CUDA support',
            '  - TensorFlow environment with DirectML support',
            '  - ONNX/CPU environment for vendor-agnostic workflows'
        ]
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def create_installation_strategy(self, nvidia_gpus: List[Dict], amd_gpus: List[Dict]) -> Dict[str, Any]:
        """
        Create optimal installation strategy for mixed environment
        """
        strategy = {
            'approach': 'vendor_separated',
            'environments': {},
            'shared_packages': [
                'numpy>=1.21.0',
                'pandas>=1.3.0',
                'pillow>=8.0.0',
                'opencv-python>=4.5.0',
                'transformers>=4.20.0',
                'diffusers>=0.20.0',
                'accelerate>=0.20.0'
            ]
        }
        
        # NVIDIA environment
        if nvidia_gpus:
            strategy['environments']['nvidia'] = {
                'name': 'nvidia_cuda_env',
                'primary_vendor': 'NVIDIA',
                'packages': [
                    'torch>=2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118',
                    'torchvision>=0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118',
                    'torchaudio>=2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118',
                    'tensorflow[and-cuda]>=2.13.0',
                    'onnxruntime-gpu>=1.15.0',
                    'xformers>=0.0.20',
                    'bitsandbytes>=0.39.0'
                ],
                'environment_variables': {
                    'CUDA_VISIBLE_DEVICES': '0,1,2,3',
                    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'
                },
                'validation_script': 'validate_nvidia_setup.py'
            }
        
        # AMD environment
        if amd_gpus:
            # Check if any AMD GPU supports ROCm
            rocm_supported = any(
                self.detect_gpu_architecture_support(gpu).get('rocm_recommended', False) 
                for gpu in amd_gpus
            )
            
            if rocm_supported:
                # Use ROCm for supported cards
                strategy['environments']['amd_rocm'] = {
                    'name': 'amd_rocm_env',
                    'primary_vendor': 'AMD',
                    'backend': 'ROCm',
                    'packages': [
                        'torch>=2.0.0+rocm5.6 --index-url https://download.pytorch.org/whl/rocm5.6',
                        'torchvision>=0.15.0+rocm5.6 --index-url https://download.pytorch.org/whl/rocm5.6',
                        'tensorflow-rocm>=2.13.0'
                    ],
                    'environment_variables': {
                        'ROCR_VISIBLE_DEVICES': '0,1,2,3',
                        'HIP_VISIBLE_DEVICES': '0,1,2,3'
                    },
                    'validation_script': 'validate_rocm_setup.py',
                    'notes': 'ROCm support for RDNA3+ GPUs'
                }
            
            # Always create DirectML environment (works on all AMD GPUs)
            strategy['environments']['amd_directml'] = {
                'name': 'amd_directml_env',
                'primary_vendor': 'AMD',
                'backend': 'DirectML',
                'packages': [
                    'torch>=2.0.0',
                    'torchvision>=0.15.0',
                    'torchaudio>=2.0.0',
                    'torch-directml>=0.2.0',
                    'tensorflow-directml>=1.15.8',
                    'onnxruntime-directml>=1.15.0'
                ],
                'environment_variables': {
                    'TF_DIRECTML_DEVICE_COUNT': str(len(amd_gpus)),
                    'DML_VISIBLE_DEVICES': '0,1,2,3,4'
                },
                'validation_script': 'validate_directml_setup.py',
                'notes': 'DirectML support for all RDNA2+ GPUs (recommended for RX 6000 series)'
            }
        
        return strategy
    
    def validate_installation_plan(self, installation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate installation plan for conflicts and compatibility issues
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for package conflicts within environments
        for env_name, env_config in installation_plan.get('environments', {}).items():
            packages = env_config.get('packages', [])
            
            # Check for conflicting PyTorch versions
            torch_variants = [pkg for pkg in packages if pkg.startswith('torch') and ('cu118' in pkg or 'rocm' in pkg)]
            if len(torch_variants) > 1:
                validation['errors'].append(f"Environment {env_name} has conflicting PyTorch variants: {torch_variants}")
                validation['valid'] = False
            
            # Check for conflicting TensorFlow versions
            tf_variants = [pkg for pkg in packages if 'tensorflow' in pkg and ('gpu' in pkg or 'directml' in pkg)]
            if len(tf_variants) > 1:
                validation['errors'].append(f"Environment {env_name} has conflicting TensorFlow variants: {tf_variants}")
                validation['valid'] = False
            
            # Check for conflicting ONNX Runtime versions
            onnx_variants = [pkg for pkg in packages if 'onnxruntime' in pkg and ('gpu' in pkg or 'directml' in pkg)]
            if len(onnx_variants) > 1:
                validation['errors'].append(f"Environment {env_name} has conflicting ONNX Runtime variants: {onnx_variants}")
                validation['valid'] = False
        
        return validation

def main():
    """Demonstrate compatibility analysis"""
    manager = GPUCompatibilityManager()
    
    # Example mixed environment
    nvidia_gpus = [{'vendor': 'NVIDIA', 'name': 'RTX 4090'}]
    amd_gpus = [
        {'vendor': 'AMD', 'name': 'Radeon RX 6800'},
        {'vendor': 'AMD', 'name': 'Radeon RX 6800 XT'}
    ]
    
    print("🔍 GPU Compatibility Analysis")
    print("=" * 40)
    
    # Analyze conflicts
    conflicts = manager.analyze_mixed_environment_conflicts(nvidia_gpus, amd_gpus)
    print(f"Environment type: {conflicts['environment_type']}")
    print(f"Total conflicts found: {len(conflicts['conflicts'])}")
    
    # Create installation strategy
    strategy = manager.create_installation_strategy(nvidia_gpus, amd_gpus)
    print(f"Recommended environments: {len(strategy['environments'])}")
    
    # Validate plan
    validation = manager.validate_installation_plan(strategy)
    print(f"Installation plan valid: {validation['valid']}")
    
    # Save analysis
    with open('gpu_compatibility_analysis.json', 'w') as f:
        json.dump({
            'conflicts': conflicts,
            'strategy': strategy,
            'validation': validation
        }, f, indent=2)
    
    print("📄 Analysis saved to gpu_compatibility_analysis.json")

if __name__ == "__main__":
    main()
