#!/usr/bin/env python3
"""
AMD GPU Detection using DXDiag
Detects AMD GPUs by parsing DirectX diagnostic information
"""

import subprocess
import tempfile
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class AMDDXDiagDetector:
    """Detect AMD GPUs using DirectX diagnostic tool"""
    
    def __init__(self):
        self.logger = None
        try:
            import logging
            self.logger = logging.getLogger(__name__)
        except ImportError:
            pass
    
    def log(self, level: str, message: str):
        """Helper logging method"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def detect_amd_gpus_dxdiag(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using DXDiag output"""
        amd_gpus = []
        
        try:
            # Create temporary file for DXDiag output
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Run DXDiag to generate system information
                self.log('info', "Running DXDiag to detect AMD GPUs...")
                result = subprocess.run(
                    ['dxdiag', '/t', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False
                )
                
                if result.returncode != 0:
                    self.log('warning', f"DXDiag exited with code {result.returncode}")
                
                # Read and parse the DXDiag output
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    dxdiag_content = f.read()
                
                # Parse AMD GPU information
                amd_gpus = self._parse_amd_gpus_from_dxdiag(dxdiag_content)
                
                if amd_gpus:
                    self.log('info', f"DXDiag detected {len(amd_gpus)} AMD GPU(s)")
                    for i, gpu in enumerate(amd_gpus):
                        self.log('info', f"  GPU {i+1}: {gpu['name']} ({gpu['memory_mb']} MB)")
                else:
                    self.log('warning', "No AMD GPUs detected via DXDiag")
                
            finally:
                # Clean up temporary file
                try:
                    Path(temp_path).unlink()
                except Exception as e:
                    self.log('warning', f"Failed to clean up temp file: {e}")
                    
        except subprocess.TimeoutExpired:
            self.log('error', "DXDiag timed out")
        except FileNotFoundError:
            self.log('error', "DXDiag not found - DirectX may not be installed")
        except Exception as e:
            self.log('error', f"Failed to run DXDiag: {e}")
        
        return amd_gpus
    
    def _parse_amd_gpus_from_dxdiag(self, dxdiag_content: str) -> List[Dict[str, Any]]:
        """Parse AMD GPU information from DXDiag output"""
        amd_gpus = []
        seen_gpus = set()  # Track unique GPUs to avoid duplicates
        
        try:
            # Look for display device sections
            lines = dxdiag_content.split('\n')
            current_gpu = {}
            in_display_section = False
            
            for line in lines:
                line = line.strip()
                
                # Check if we're entering a display device section
                if 'Display Devices' in line:
                    in_display_section = True
                    continue
                elif line.startswith('Sound Devices') or line.startswith('Sound Capture'):
                    in_display_section = False
                    continue
                
                if not in_display_section:
                    continue
                
                # Parse GPU information
                if line.startswith('Card name:'):
                    # Save previous GPU if it was AMD
                    if current_gpu and self._is_amd_gpu(current_gpu.get('name', '')):
                        gpu_signature = f"{current_gpu['name']}_{current_gpu.get('memory_mb', 0)}"
                        if gpu_signature not in seen_gpus:
                            amd_gpus.append(current_gpu)
                            seen_gpus.add(gpu_signature)
                    
                    # Start new GPU
                    gpu_name = line.split(':', 1)[1].strip()
                    current_gpu = {
                        'name': gpu_name,
                        'vendor': 'AMD',
                        'memory_mb': 0,
                        'driver_version': 'Unknown',
                        'chip_type': 'Unknown'
                    }
                
                elif line.startswith('Chip type:') and current_gpu:
                    current_gpu['chip_type'] = line.split(':', 1)[1].strip()
                
                elif line.startswith('Display Memory:') and current_gpu:
                    # Parse memory (e.g., "32655 MB")
                    memory_str = line.split(':', 1)[1].strip()
                    memory_match = re.search(r'(\d+)\s*MB', memory_str)
                    if memory_match:
                        current_gpu['memory_mb'] = int(memory_match.group(1))
                
                elif line.startswith('Driver Name:') and current_gpu:
                    # Extract driver version from driver name
                    driver_info = line.split(':', 1)[1].strip()
                    # Look for version pattern in the driver path
                    version_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', driver_info)
                    if version_match:
                        current_gpu['driver_version'] = version_match.group(1)
            
            # Don't forget the last GPU
            if current_gpu and self._is_amd_gpu(current_gpu.get('name', '')):
                gpu_signature = f"{current_gpu['name']}_{current_gpu.get('memory_mb', 0)}"
                if gpu_signature not in seen_gpus:
                    amd_gpus.append(current_gpu)
                    seen_gpus.add(gpu_signature)
            
            # Add metadata to each GPU
            for gpu in amd_gpus:
                gpu.update({
                    'id': f"amd_dxdiag_{len(amd_gpus)}_" + gpu['name'].replace(' ', '_').lower(),
                    'architecture': self._get_amd_architecture(gpu['name']),
                    'memory_gb': round(gpu['memory_mb'] / 1024, 1),
                    'compute_capability': self._get_amd_compute_capability(gpu['name']),
                    'directml_support': True,  # AMD RDNA2+ supports DirectML
                    'metadata': {
                        'detection_method': 'dxdiag',
                        'collection_time': datetime.now().isoformat(),
                        'chip_type': gpu['chip_type']
                    }
                })
            
        except Exception as e:
            self.log('error', f"Failed to parse DXDiag output: {e}")
        
        return amd_gpus
    
    def _is_amd_gpu(self, name: str) -> bool:
        """Check if GPU name indicates an AMD GPU"""
        amd_indicators = ['amd', 'radeon', 'rx ', 'vega', 'navi']
        name_lower = name.lower()
        return any(indicator in name_lower for indicator in amd_indicators)
    
    def _get_amd_architecture(self, name: str) -> str:
        """Determine AMD GPU architecture from name"""
        name_lower = name.lower()
        
        if 'rx 6' in name_lower or '6800' in name_lower or '6900' in name_lower:
            return 'RDNA2'
        elif 'rx 7' in name_lower:
            return 'RDNA3'
        elif 'rx 5' in name_lower:
            return 'RDNA'
        elif 'vega' in name_lower:
            return 'GCN5'
        elif 'rx 4' in name_lower or 'rx 5500' in name_lower:
            return 'GCN4'
        else:
            return 'Unknown'
    
    def _get_amd_compute_capability(self, name: str) -> str:
        """Get AMD compute capability equivalent"""
        arch = self._get_amd_architecture(name)
        
        compute_map = {
            'RDNA3': 'gfx1100',
            'RDNA2': 'gfx1030',
            'RDNA': 'gfx1010',
            'GCN5': 'gfx906',
            'GCN4': 'gfx803'
        }
        
        return compute_map.get(arch, 'Unknown')

def main():
    """Test the AMD DXDiag detector"""
    print("🔍 AMD DXDiag GPU Detection Test")
    print("=" * 50)
    
    detector = AMDDXDiagDetector()
    gpus = detector.detect_amd_gpus_dxdiag()
    
    if gpus:
        print(f"✅ Detected {len(gpus)} AMD GPU(s):")
        for i, gpu in enumerate(gpus, 1):
            print(f"\n  GPU {i}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_gb']} GB ({gpu['memory_mb']} MB)")
            print(f"    Architecture: {gpu['architecture']}")
            print(f"    Compute: {gpu['compute_capability']}")
            print(f"    DirectML: {gpu['directml_support']}")
            print(f"    Driver: {gpu['driver_version']}")
        
        # Save results
        with open('amd_dxdiag_results.json', 'w') as f:
            json.dump(gpus, f, indent=2)
        print(f"\n💾 Results saved to amd_dxdiag_results.json")
    else:
        print("❌ No AMD GPUs detected")
    
    print("\n🎉 Detection complete!")

if __name__ == "__main__":
    main()
