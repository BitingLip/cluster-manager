"""
Simple GPU Detector Fallback
Basic GPU detection functionality when the main gpu_detector module is not available
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class GPUDetector:
    """
    Basic GPU detector for fallback scenarios
    """
    
    def __init__(self):
        self.detected_gpus = []
        
    def detect_all_gpus(self) -> Dict[str, Any]:
        """
        Detect all available GPUs using basic PyTorch detection
        """
        result = {
            'nvidia_gpus': self._detect_nvidia_gpus(),
            'amd_gpus': self._detect_amd_gpus(),
            'total_gpus': 0,
            'detection_method': 'fallback_basic'
        }
        
        result['total_gpus'] = len(result['nvidia_gpus']) + len(result['amd_gpus'])
        
        logger.info(f"Basic GPU detection found {result['total_gpus']} GPUs")
        logger.info(f"NVIDIA: {len(result['nvidia_gpus'])}, AMD: {len(result['amd_gpus'])}")
        
        return result
    
    def _detect_nvidia_gpus(self) -> List[Dict[str, Any]]:
        """Detect NVIDIA GPUs using PyTorch CUDA"""
        nvidia_gpus = []
        
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    nvidia_gpus.append({
                        'id': i,
                        'name': props.name,
                        'vendor': 'NVIDIA',
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}",
                        'device_string': f'cuda:{i}',
                        'framework': 'torch_cuda'
                    })
                logger.info(f"Found {device_count} NVIDIA GPU(s)")
        except ImportError:
            logger.warning("PyTorch not available for NVIDIA detection")
        except Exception as e:
            logger.error(f"Error detecting NVIDIA GPUs: {e}")
            
        return nvidia_gpus
    
    def _detect_amd_gpus(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using DirectML"""
        amd_gpus = []
        
        try:
            import torch_directml  # type: ignore
            device_count = torch_directml.device_count()
            for i in range(device_count):
                amd_gpus.append({
                    'id': i,
                    'name': f'AMD DirectML Device {i}',
                    'vendor': 'AMD',
                    'device_string': f'privateuseone:{i}',
                    'framework': 'torch_directml',
                    'requires_system_python': True
                })
            logger.info(f"Found {device_count} AMD GPU(s) via DirectML")
        except ImportError:
            logger.warning("DirectML not available for AMD detection")
        except Exception as e:
            logger.error(f"Error detecting AMD GPUs: {e}")
            
        return amd_gpus
    
    def get_gpu_summary(self) -> str:
        """Get a summary of detected GPUs"""
        gpus = self.detect_all_gpus()
        
        summary = f"GPU Detection Summary:\n"
        summary += f"Total GPUs: {gpus['total_gpus']}\n"
        summary += f"NVIDIA GPUs: {len(gpus['nvidia_gpus'])}\n"
        summary += f"AMD GPUs: {len(gpus['amd_gpus'])}\n"
        summary += f"Detection Method: {gpus['detection_method']}\n"
        
        if gpus['nvidia_gpus']:
            summary += "\nNVIDIA GPUs:\n"
            for gpu in gpus['nvidia_gpus']:
                summary += f"  - {gpu['name']} ({gpu['memory_gb']:.1f}GB)\n"
                
        if gpus['amd_gpus']:
            summary += "\nAMD GPUs:\n"
            for gpu in gpus['amd_gpus']:
                summary += f"  - {gpu['name']}\n"
                
        return summary


# Convenience functions for backward compatibility
def detect_nvidia_gpus() -> List[Dict[str, Any]]:
    """Quick function to detect NVIDIA GPUs"""
    detector = GPUDetector()
    return detector._detect_nvidia_gpus()


def detect_amd_gpus() -> List[Dict[str, Any]]:
    """Quick function to detect AMD GPUs"""
    detector = GPUDetector()
    return detector._detect_amd_gpus()


def detect_all_gpus() -> Dict[str, Any]:
    """Quick function to detect all GPUs"""
    detector = GPUDetector()
    return detector.detect_all_gpus()


if __name__ == "__main__":
    print("🔍 Basic GPU Detection Test")
    print("=" * 40)
    
    detector = GPUDetector()
    print(detector.get_gpu_summary())
