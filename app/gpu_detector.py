"""
Enhanced GPU Detection Module for Cluster Manager
Supports both NVIDIA and AMD GPUs with AI capability assessment
"""

import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class GPUDetector:
    """Enhanced GPU detection for both NVIDIA and AMD GPUs"""
    
    def __init__(self):
        self.supported_vendors = ['NVIDIA', 'AMD']
        
    def detect_all_gpus(self) -> List[Dict[str, Any]]:
        """Detect all available GPUs from supported vendors"""
        all_gpus = []
        
        # Detect NVIDIA GPUs
        nvidia_gpus = self._detect_nvidia_gpus()
        all_gpus.extend(nvidia_gpus)
        
        # Detect AMD GPUs
        amd_gpus = self._detect_amd_gpus()
        all_gpus.extend(amd_gpus)
        
        if all_gpus:
            logger.info(f"Detected {len(all_gpus)} GPU(s) total")
        else:
            logger.info("No GPUs detected")
            
        return all_gpus
    
    def _detect_nvidia_gpus(self) -> List[Dict[str, Any]]:
        """Detect NVIDIA GPUs using NVML"""
        nvidia_gpus = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU name
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total_mb = mem_info.total // (1024 * 1024)
                memory_used_mb = mem_info.used // (1024 * 1024)
                memory_available_mb = (mem_info.total - mem_info.used) // (1024 * 1024)
                
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent = util.gpu
                
                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Get power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power_usage = None
                
                # Get compute capability
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "Unknown"
                
                # Assess AI capabilities
                ai_capabilities = self._assess_nvidia_ai_capabilities(name, compute_capability, memory_total_mb)
                
                gpu_info = {
                    'index': i,
                    'vendor': 'NVIDIA',
                    'name': name,
                    'memory_total_mb': memory_total_mb,
                    'memory_used_mb': memory_used_mb,
                    'memory_available_mb': memory_available_mb,
                    'utilization_percent': utilization_percent,
                    'temperature_celsius': temperature,
                    'power_usage_watts': power_usage,
                    'compute_capability': compute_capability,
                    'ai_capabilities': ai_capabilities,
                    'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                    'framework_support': ['CUDA', 'DirectML', 'OpenCL'],                    'metadata': {
                        'collection_time': datetime.now().isoformat(),
                        'detection_method': 'pynvml',
                        'architecture': self._get_nvidia_architecture(name)
                    }
                }
                
                nvidia_gpus.append(gpu_info)
            
            if nvidia_gpus:
                logger.info(f"Detected {len(nvidia_gpus)} NVIDIA GPU(s)")
            
        except ImportError:
            logger.debug("pynvml not available, skipping NVIDIA GPU detection")
        except Exception as e:
            logger.warning(f"Failed to detect NVIDIA GPUs: {str(e)}")
        
        return nvidia_gpus
    
    def _detect_amd_gpus(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using multiple detection methods"""
        amd_gpus = []
        
        # Method 1: WMI (Windows only)
        if sys.platform == 'win32':
            wmi_gpus = self._detect_amd_gpus_wmi()
            amd_gpus.extend(wmi_gpus)
        
        # Method 2: OpenCL (cross-platform)
        opencl_gpus = self._detect_amd_gpus_opencl()
        amd_gpus.extend(opencl_gpus)
        
        # Remove duplicates based on name, keeping most detailed info
        unique_gpus = {}
        for gpu in amd_gpus:
            key = gpu['name']
            if key not in unique_gpus or len(gpu) > len(unique_gpus[key]):
                unique_gpus[key] = gpu
        
        final_amd_gpus = list(unique_gpus.values())
        
        # Re-index the GPUs
        for i, gpu in enumerate(final_amd_gpus):
            gpu['index'] = i
        
        if final_amd_gpus:
            logger.info(f"Detected {len(final_amd_gpus)} AMD GPU(s)")
        
        return final_amd_gpus
    
    def _detect_amd_gpus_wmi(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using Windows WMI"""
        amd_gpus = []
        
        try:
            import wmi
            c = wmi.WMI()
            
            for gpu in c.Win32_VideoController():
                if gpu.Name and ('AMD' in gpu.Name or 'Radeon' in gpu.Name or 'ATI' in gpu.Name):                    # Get available memory - use correct specs for known GPUs
                    memory_mb = 0
                    if gpu.AdapterRAM:
                        memory_mb = gpu.AdapterRAM // (1024 * 1024)
                    
                    # Apply correct memory specs for known AMD GPUs
                    correct_memory = self._get_correct_amd_memory(gpu.Name)
                    if correct_memory > 0:
                        memory_mb = correct_memory
                    
                    # Assess AI capabilities
                    ai_capabilities = self._assess_amd_ai_capabilities(gpu.Name, memory_mb)
                    
                    gpu_info = {
                        'index': len(amd_gpus),
                        'vendor': 'AMD',
                        'name': gpu.Name,
                        'memory_total_mb': memory_mb,
                        'memory_used_mb': 0,  # WMI doesn't provide current usage
                        'memory_available_mb': memory_mb,
                        'utilization_percent': 0,  # WMI doesn't provide utilization
                        'temperature_celsius': None,
                        'power_usage_watts': None,
                        'driver_version': gpu.DriverVersion or 'Unknown',
                        'ai_capabilities': ai_capabilities,
                        'framework_support': self._get_amd_framework_support(gpu.Name),                        'metadata': {                            'collection_time': datetime.now().isoformat(),
                            'detection_method': 'wmi',
                            'architecture': self._get_amd_architecture(gpu.Name),
                            'pnp_device_id': gpu.PNPDeviceID or 'Unknown'
                        }
                    }
                    amd_gpus.append(gpu_info)
                    
        except ImportError:
            logger.debug("WMI not available, trying PowerShell method")
            
            # Fallback: PowerShell WMI method
            try:
                import subprocess
                import json
                
                ps_command = '''
                Get-WmiObject Win32_VideoController | 
                Where-Object {$_.Name -like "*AMD*" -or $_.Name -like "*Radeon*"} | 
                Select-Object Name, AdapterRAM, PNPDeviceID, Status, DriverVersion | 
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
                    
                    for gpu in gpu_data:
                        if gpu.get('Name'):
                            # Convert memory from bytes to MB
                            memory_bytes = gpu.get('AdapterRAM', 0)
                            memory_mb = memory_bytes // (1024 * 1024) if memory_bytes else 0
                            
                            # Assess AI capabilities
                            ai_capabilities = self._assess_amd_ai_capabilities(gpu['Name'], memory_mb)
                            
                            gpu_info = {
                                'index': len(amd_gpus),
                                'vendor': 'AMD',
                                'name': gpu['Name'],
                                'memory_total_mb': memory_mb,
                                'memory_used_mb': 0,
                                'memory_available_mb': memory_mb,
                                'utilization_percent': 0,
                                'temperature_celsius': None,
                                'power_usage_watts': None,
                                'driver_version': gpu.get('DriverVersion', 'Unknown'),
                                'ai_capabilities': ai_capabilities,
                                'framework_support': self._get_amd_framework_support(gpu['Name']),
                                'metadata': {
                                    'collection_time': datetime.now().isoformat(),
                                    'detection_method': 'powershell_wmi',
                                    'architecture': self._get_amd_architecture(gpu['Name']),
                                    'pnp_device_id': gpu.get('PNPDeviceID', 'Unknown')
                                }
                            }
                            amd_gpus.append(gpu_info)
                            
            except Exception as ps_error:
                logger.debug(f"PowerShell WMI method failed: {ps_error}")
                
        except Exception as e:
            logger.warning(f"Failed to detect AMD GPUs via WMI: {str(e)}")
        
        return amd_gpus
    
    def _detect_amd_gpus_opencl(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using OpenCL"""
        amd_gpus = []
        
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            for platform in platforms:
                if any(keyword in platform.name for keyword in ['AMD', 'Advanced Micro Devices']):
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    
                    for device in devices:
                        memory_mb = device.global_mem_size // (1024 * 1024)
                        
                        # Assess AI capabilities
                        ai_capabilities = self._assess_amd_ai_capabilities(device.name, memory_mb)
                        
                        gpu_info = {
                            'index': len(amd_gpus),
                            'vendor': 'AMD',
                            'name': device.name,
                            'memory_total_mb': memory_mb,
                            'memory_used_mb': 0,  # OpenCL doesn't provide usage
                            'memory_available_mb': memory_mb,
                            'utilization_percent': 0,
                            'temperature_celsius': None,
                            'power_usage_watts': None,
                            'compute_units': device.max_compute_units,
                            'max_work_group_size': device.max_work_group_size,
                            'ai_capabilities': ai_capabilities,
                            'framework_support': self._get_amd_framework_support(device.name),
                            'metadata': {
                                'collection_time': datetime.now().isoformat(),
                                'detection_method': 'opencl',
                                'architecture': self._get_amd_architecture(device.name),
                                'opencl_version': device.version,
                                'platform': platform.name
                            }
                        }
                        amd_gpus.append(gpu_info)
            
        except ImportError:
            logger.debug("PyOpenCL not available for AMD GPU detection")
        except Exception as e:
            logger.debug(f"OpenCL AMD GPU detection failed: {str(e)}")
        
        return amd_gpus
    
    def _assess_nvidia_ai_capabilities(self, name: str, compute_capability: str, memory_mb: int) -> Dict[str, Any]:
        """Assess AI capabilities of NVIDIA GPU"""
        capabilities = {
            'tensor_cores': False,
            'fp16_support': False,
            'int8_support': False,
            'suitable_for_inference': False,
            'suitable_for_training': False,
            'recommended_models': [],
            'performance_tier': 'low'
        }
        
        # Parse compute capability
        try:
            major = int(compute_capability.split('.')[0])
            minor = int(compute_capability.split('.')[1])
        except:
            major, minor = 0, 0
        
        # Tensor cores available from compute capability 7.0+
        if major >= 8:  # Ampere and newer
            capabilities.update({
                'tensor_cores': True,
                'fp16_support': True,
                'int8_support': True,
                'performance_tier': 'high'
            })
        elif major >= 7:  # Turing
            capabilities.update({
                'tensor_cores': True,
                'fp16_support': True,
                'int8_support': True,
                'performance_tier': 'medium-high'
            })
        elif major >= 6:  # Pascal
            capabilities.update({
                'fp16_support': True,
                'performance_tier': 'medium'
            })
        
        # Memory-based recommendations
        if memory_mb >= 24000:  # 24GB+
            capabilities.update({
                'suitable_for_training': True,
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-65B-4bit', 'Stable Diffusion XL', 'GPT-3.5 equivalent', 'Large model training']
            })
        elif memory_mb >= 16000:  # 16GB+
            capabilities.update({
                'suitable_for_training': True,
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-30B-4bit', 'Stable Diffusion 2.1', 'Fine-tuning 13B models', 'Medium model training']
            })
        elif memory_mb >= 12000:  # 12GB+
            capabilities.update({
                'suitable_for_training': True,
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-13B-4bit', 'Stable Diffusion XL (optimized)', 'Small model training']
            })
        elif memory_mb >= 8000:  # 8GB+
            capabilities.update({
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-13B-4bit', 'Stable Diffusion 1.5', 'Small model training']
            })
        elif memory_mb >= 4000:  # 4GB+
            capabilities.update({
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-7B-4bit', 'Stable Diffusion (optimized)', 'Small CNNs']
            })
        
        return capabilities
    
    def _assess_amd_ai_capabilities(self, name: str, memory_mb: int) -> Dict[str, Any]:
        """Assess AI capabilities of AMD GPU based on architecture and specs"""
        capabilities = {
            'directml_support': True,  # All DX12 AMD GPUs support DirectML
            'fp16_support': False,
            'quantization_support': False,
            'driver_optimized': False,
            'ai_accelerators': False,
            'suitable_for_inference': False,
            'suitable_for_training': False,
            'recommended_models': [],
            'performance_tier': 'low',
            'architecture_notes': ''
        }
        
        # Determine architecture and capabilities
        arch = self._get_amd_architecture(name)
        
        if 'RDNA3' in arch:
            capabilities.update({
                'fp16_support': True,
                'quantization_support': True,
                'driver_optimized': True,
                'ai_accelerators': True,
                'performance_tier': 'high',
                'architecture_notes': 'Latest AMD architecture with AI accelerators and optimized DirectML drivers (23.40.27.06+)'
            })
        elif 'RDNA2' in arch:
            capabilities.update({
                'fp16_support': True,
                'quantization_support': True,
                'driver_optimized': True,
                'performance_tier': 'medium-high',
                'architecture_notes': 'Good AI performance with DirectML optimizations and AWQ quantization support'
            })
        elif 'RDNA1' in arch:
            capabilities.update({
                'fp16_support': True,
                'quantization_support': True,
                'driver_optimized': True,
                'performance_tier': 'medium',
                'architecture_notes': 'Moderate AI performance with DirectML optimizations, supported by AI drivers'
            })
        elif 'Vega' in arch:
            capabilities.update({
                'fp16_support': True,  # Vega has "rapid packed math"
                'quantization_support': False,
                'driver_optimized': False,
                'performance_tier': 'low-medium',
                'architecture_notes': 'Legacy architecture with basic DirectML support, FP16 rapid packed math, no recent AI optimizations'
            })
        elif 'Polaris' in arch:
            capabilities.update({
                'fp16_support': False,  # FP32 only
                'quantization_support': False,
                'driver_optimized': False,
                'performance_tier': 'low',
                'architecture_notes': 'Legacy architecture with basic DirectML support, FP32 only, no AI optimizations'
            })
        
        # Memory-based recommendations (adjusted for AMD DirectML performance)
        if memory_mb >= 20000:  # 20GB+ (RX 7900 XT/XTX)
            capabilities.update({
                'suitable_for_training': True,
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-30B-4bit', 'Stable Diffusion XL', 'Large model fine-tuning', 'Multiple concurrent models']
            })
        elif memory_mb >= 16000:  # 16GB (RX 6800 XT, Radeon VII)
            capabilities.update({
                'suitable_for_training': True,
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-13B-4bit', 'Stable Diffusion 2.1', 'Medium model training', 'LoRA fine-tuning']
            })
        elif memory_mb >= 12000:  # 12GB (RX 6700 XT)
            capabilities.update({
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-13B-4bit', 'Stable Diffusion XL (optimized)', 'Small model training']
            })
        elif memory_mb >= 8000:  # 8GB (RX 580, RX 5700, RX 6600)
            capabilities.update({
                'suitable_for_inference': True,
                'recommended_models': ['LLaMA-7B-4bit', 'Stable Diffusion 1.5', 'Phi-3', 'Small model inference']
            })
        elif memory_mb >= 4000:  # 4GB (RX 580 4GB, RX 470)
            capabilities.update({
                'suitable_for_inference': True,
                'recommended_models': ['Phi-3 mini', 'Stable Diffusion (low VRAM mode)', 'Basic CNNs', 'Small transformers']
            })
        
        return capabilities
    
    def _get_nvidia_architecture(self, name: str) -> str:
        """Determine NVIDIA GPU architecture from name"""
        name_upper = name.upper()
        
        if any(x in name_upper for x in ['RTX 40', 'RTX 4']):
            return 'Ada Lovelace (RTX 40 series)'
        elif any(x in name_upper for x in ['RTX 30', 'RTX 3']):
            return 'Ampere (RTX 30 series)'
        elif any(x in name_upper for x in ['RTX 20', 'RTX 2']):
            return 'Turing (RTX 20 series)'
        elif 'GTX 16' in name_upper:
            return 'Turing (GTX 16 series)'
        elif 'GTX 10' in name_upper:
            return 'Pascal (GTX 10 series)'
        elif 'GTX 9' in name_upper:
            return 'Maxwell (GTX 9 series)'
        elif any(x in name_upper for x in ['GTX 7', 'GTX 6']):
            return 'Kepler'
        else:
            return 'Unknown NVIDIA Architecture'
    
    def _get_amd_architecture(self, name: str) -> str:
        """Determine AMD GPU architecture from name"""
        name_upper = name.upper()
        
        # RDNA3 - RX 7000 series
        if any(x in name_upper for x in ['RX 7', '7900', '7800', '7700', '7600']):
            return 'RDNA3 (RX 7000 series)'
        
        # RDNA2 - RX 6000 series  
        elif any(x in name_upper for x in ['RX 6', '6900', '6800', '6700', '6600', '6500', '6400']):
            return 'RDNA2 (RX 6000 series)'
        
        # RDNA1 - RX 5000 series
        elif any(x in name_upper for x in ['RX 5', '5700', '5600', '5500']) and not any(x in name_upper for x in ['580', '570']):
            return 'RDNA1 (RX 5000 series)'
        
        # Vega architecture
        elif any(x in name_upper for x in ['VEGA', 'RADEON VII']):
            return 'Vega (GCN 5)'
        
        # Polaris - RX 400/500 series
        elif any(x in name_upper for x in ['RX 5', 'RX 4']) and any(x in name_upper for x in ['80', '70', '60', '50']):
            return 'Polaris (GCN 4)'
        
        # Older GCN
        elif any(x in name_upper for x in ['R9', 'R7', 'R5']):
            return 'GCN (Legacy)'
        
        else:
            return 'Unknown AMD Architecture'
    
    def _get_amd_framework_support(self, name: str) -> List[str]:
        """Get supported frameworks for AMD GPU"""
        frameworks = ['DirectML']  # All modern AMD GPUs support DirectML
        
        arch = self._get_amd_architecture(name)
        
        # Add framework support based on architecture
        if 'RDNA' in arch:
            frameworks.extend(['PyTorch-DirectML', 'ONNX-DirectML', 'TensorFlow-DirectML'])
        
        # OpenCL support is universal for AMD
        frameworks.append('OpenCL')
        
        # ROCm support (Linux, newer cards)
        if any(x in arch for x in ['RDNA2', 'RDNA3']):
            frameworks.append('ROCm (Linux)')
        
        # Add specialized tools
        frameworks.extend(['llama.cpp (OpenCL)', 'Olive (DirectML)'])
        
        return frameworks
    
    def _get_correct_amd_memory(self, gpu_name: str) -> int:
        """
        Get correct memory specification for known AMD GPUs
        WMI often reports incorrect memory values for AMD GPUs
        """
        gpu_name_lower = gpu_name.lower()
        
        # RX 6800 and RX 6800 XT both have 16GB VRAM
        if '6800' in gpu_name_lower:
            return 16384  # 16GB in MB
        
        # RX 6900 XT has 16GB VRAM
        if '6900' in gpu_name_lower:
            return 16384
        
        # RX 6700 XT has 12GB VRAM
        if '6700 xt' in gpu_name_lower:
            return 12288
        
        # RX 6700 has 10GB VRAM
        if '6700' in gpu_name_lower and 'xt' not in gpu_name_lower:
            return 10240
        
        # RX 6600 XT has 8GB VRAM
        if '6600 xt' in gpu_name_lower:
            return 8192
        
        # RX 6600 has 8GB VRAM
        if '6600' in gpu_name_lower:
            return 8192
        
        # RX 6500 XT has 4GB VRAM
        if '6500' in gpu_name_lower:
            return 4096
        
        # RX 6400 has 4GB VRAM
        if '6400' in gpu_name_lower:
            return 4096
        
        # RDNA3 GPUs (RX 7000 series)
        if '7900 xtx' in gpu_name_lower:
            return 24576  # 24GB
        
        if '7900 xt' in gpu_name_lower:
            return 20480  # 20GB
        
        if '7800 xt' in gpu_name_lower:
            return 16384  # 16GB
        
        if '7700 xt' in gpu_name_lower:
            return 12288  # 12GB
        
        if '7600' in gpu_name_lower:
            return 8192   # 8GB
        
        # Return 0 if unknown - let WMI value be used
        return 0
