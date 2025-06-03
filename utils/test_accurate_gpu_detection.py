#!/usr/bin/env python3
"""
Accurate AMD GPU Detection Test
Specifically designed for 4x RX 6800 + 1x RX 6800 XT setup
"""

import sys
import os
import json
import subprocess
import re
from collections import Counter
from typing import List, Dict, Any

# Add the parent directory to the path to import gpu_detector
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def get_accurate_amd_gpu_count() -> Dict[str, Any]:
    """
    Get accurate AMD GPU count using multiple detection methods
    """
    results = {
        'powershell_wmi': [],
        'dxdiag': [],
        'wmic': [],
        'expected': {
            'rx_6800': 4,
            'rx_6800_xt': 1,
            'total': 5
        }
    }
    
    # Method 1: PowerShell WMI with detailed memory detection
    print("🔍 Method 1: PowerShell WMI Detection")
    try:
        ps_command = '''
        Get-WmiObject Win32_VideoController | 
        Where-Object {$_.Name -like "*AMD*" -and $_.Name -like "*Radeon*" -and $_.Name -like "*RX*"} | 
        Select-Object Name, AdapterRAM, PNPDeviceID, DriverVersion, Status |
        ForEach-Object {
            $memoryGB = if ($_.AdapterRAM) { [math]::Round($_.AdapterRAM / 1GB, 2) } else { 0 }
            [PSCustomObject]@{
                Name = $_.Name
                MemoryGB = $memoryGB
                MemoryMB = if ($_.AdapterRAM) { [math]::Round($_.AdapterRAM / 1MB, 0) } else { 0 }
                PNPDeviceID = $_.PNPDeviceID
                DriverVersion = $_.DriverVersion
                Status = $_.Status
            }
        } | ConvertTo-Json
        '''
        
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_data = json.loads(result.stdout)
            if isinstance(gpu_data, dict):
                gpu_data = [gpu_data]
            
            results['powershell_wmi'] = gpu_data
            print(f"   Found {len(gpu_data)} GPUs via PowerShell")
            
            for gpu in gpu_data:
                print(f"   - {gpu['Name']}: {gpu['MemoryGB']}GB")
                
    except Exception as e:
        print(f"   ❌ PowerShell WMI failed: {e}")
    
    # Method 2: WMIC Command Line
    print("\n🔍 Method 2: WMIC Detection")
    try:
        result = subprocess.run(
            ['wmic', 'path', 'win32_videocontroller', 'where', 'name like "%AMD%" and name like "%Radeon%" and name like "%RX%"', 'get', 'name,adapterram,pnpdeviceid', '/format:csv'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            headers = None
            wmic_gpus = []
            
            for line in lines:
                if not headers and 'AdapterRAM' in line:
                    headers = [h.strip() for h in line.split(',')]
                elif headers and line and ',' in line:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) >= len(headers):
                        gpu_dict = dict(zip(headers, values))
                        if gpu_dict.get('Name') and 'RX' in gpu_dict['Name']:
                            memory_bytes = int(gpu_dict.get('AdapterRAM', 0)) if gpu_dict.get('AdapterRAM', '').isdigit() else 0
                            memory_gb = round(memory_bytes / (1024**3), 2) if memory_bytes else 0
                            
                            wmic_gpus.append({
                                'Name': gpu_dict['Name'],
                                'MemoryGB': memory_gb,
                                'PNPDeviceID': gpu_dict.get('PNPDeviceID', 'Unknown')
                            })
            
            results['wmic'] = wmic_gpus
            print(f"   Found {len(wmic_gpus)} GPUs via WMIC")
            
            for gpu in wmic_gpus:
                print(f"   - {gpu['Name']}: {gpu['MemoryGB']}GB")
                
    except Exception as e:
        print(f"   ❌ WMIC failed: {e}")
    
    # Method 3: DxDiag with better parsing
    print("\n🔍 Method 3: Enhanced DxDiag Detection")
    try:
        # Run dxdiag
        subprocess.run(["dxdiag", "/t", "temp_dxdiag.txt"], timeout=60)
        
        with open("temp_dxdiag.txt", "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Parse AMD RX 6800 series GPUs specifically
        card_pattern = r"Card name:\s*(AMD Radeon RX 68[0-9]+[^\n]*)"
        memory_pattern = r"Display Memory:\s*(\d+)\s*MB"
        
        cards = re.findall(card_pattern, content, re.IGNORECASE)
        
        dxdiag_gpus = []
        for card in cards:
            # Find the display memory for this card
            card_pos = content.find(f"Card name: {card}")
            if card_pos != -1:
                # Look for memory info in the next 1000 characters
                section = content[card_pos:card_pos + 1000]
                memory_match = re.search(memory_pattern, section)
                memory_mb = int(memory_match.group(1)) if memory_match else 0
                memory_gb = round(memory_mb / 1024, 2)
                
                dxdiag_gpus.append({
                    'Name': card.strip(),
                    'MemoryGB': memory_gb,
                    'MemoryMB': memory_mb
                })
        
        results['dxdiag'] = dxdiag_gpus
        print(f"   Found {len(dxdiag_gpus)} GPUs via DxDiag")
        
        for gpu in dxdiag_gpus:
            print(f"   - {gpu['Name']}: {gpu['MemoryGB']}GB")
        
        # Clean up
        try:
            os.remove("temp_dxdiag.txt")
        except:
            pass
            
    except Exception as e:
        print(f"   ❌ DxDiag failed: {e}")
    
    return results

def analyze_gpu_detection_accuracy(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the accuracy of GPU detection against expected configuration
    """
    expected = results['expected']
    analysis = {
        'method_accuracy': {},
        'consensus': {},
        'recommendations': []
    }
    
    print("\n📊 Analysis: GPU Detection Accuracy")
    print("=" * 50)
    
    for method, gpus in results.items():
        if method == 'expected' or not gpus:
            continue
            
        # Count GPU models
        gpu_counts = Counter()
        total_memory_per_gpu = {}
        
        for gpu in gpus:
            name = gpu['Name']
            memory = gpu.get('MemoryGB', 0)
            
            if 'RX 6800 XT' in name:
                gpu_counts['RX 6800 XT'] += 1
                total_memory_per_gpu.setdefault('RX 6800 XT', []).append(memory)
            elif 'RX 6800' in name:
                gpu_counts['RX 6800'] += 1
                total_memory_per_gpu.setdefault('RX 6800', []).append(memory)
        
        # Calculate accuracy
        rx_6800_accuracy = min(100, (gpu_counts.get('RX 6800', 0) / expected['rx_6800']) * 100)
        rx_6800_xt_accuracy = min(100, (gpu_counts.get('RX 6800 XT', 0) / expected['rx_6800_xt']) * 100)
        total_accuracy = min(100, (sum(gpu_counts.values()) / expected['total']) * 100)
        
        analysis['method_accuracy'][method] = {
            'detected_rx_6800': gpu_counts.get('RX 6800', 0),
            'detected_rx_6800_xt': gpu_counts.get('RX 6800 XT', 0),
            'total_detected': sum(gpu_counts.values()),
            'rx_6800_accuracy': round(rx_6800_accuracy, 1),
            'rx_6800_xt_accuracy': round(rx_6800_xt_accuracy, 1),
            'total_accuracy': round(total_accuracy, 1),
            'memory_per_gpu': total_memory_per_gpu
        }
        
        print(f"\n🎯 {method.upper()} Method:")
        print(f"   RX 6800: {gpu_counts.get('RX 6800', 0)}/4 ({rx_6800_accuracy:.1f}% accuracy)")
        print(f"   RX 6800 XT: {gpu_counts.get('RX 6800 XT', 0)}/1 ({rx_6800_xt_accuracy:.1f}% accuracy)")
        print(f"   Total: {sum(gpu_counts.values())}/5 ({total_accuracy:.1f}% accuracy)")
        
        # Memory analysis
        for model, memories in total_memory_per_gpu.items():
            if memories:
                avg_memory = sum(memories) / len(memories)
                print(f"   {model} Memory: {avg_memory:.1f}GB avg")
    
    # Find best method
    best_method = None
    best_accuracy = 0
    
    for method, accuracy_data in analysis['method_accuracy'].items():
        if accuracy_data['total_accuracy'] > best_accuracy:
            best_accuracy = accuracy_data['total_accuracy']
            best_method = method
    
    analysis['best_method'] = best_method
    analysis['best_accuracy'] = best_accuracy
    
    print(f"\n🏆 Best Detection Method: {best_method} ({best_accuracy:.1f}% accuracy)")
    
    # Recommendations
    if best_accuracy < 100:
        analysis['recommendations'].append("Consider using multiple detection methods for redundancy")
    
    if best_method == 'dxdiag':
        analysis['recommendations'].append("DxDiag provides most accurate memory information")
    elif best_method == 'powershell_wmi':
        analysis['recommendations'].append("PowerShell WMI is most reliable for GPU enumeration")
    
    return analysis

def test_gpu_detector_integration():
    """
    Test the actual GPU detector integration
    """
    print("\n🔧 Testing GPU Detector Integration")
    print("=" * 40)
    
    try:
        from app.gpu_detector import GPUDetector
        
        detector = GPUDetector()
        gpus = detector.detect_gpus()
        
        amd_gpus = [gpu for gpu in gpus if gpu.get('vendor') == 'AMD']
        
        print(f"GPU Detector found {len(amd_gpus)} AMD GPUs:")
        
        for i, gpu in enumerate(amd_gpus, 1):
            print(f"   {i}. {gpu['name']}")
            print(f"      Memory: {gpu.get('memory_total_mb', 0)/1024:.1f}GB")
            print(f"      Method: {gpu.get('metadata', {}).get('detection_method', 'Unknown')}")
            
        return len(amd_gpus)
        
    except Exception as e:
        print(f"❌ GPU Detector integration failed: {e}")
        return 0

def main():
    """
    Main test function
    """
    print("🚀 Accurate AMD GPU Detection Test")
    print("=" * 60)
    print("Expected Configuration: 4x RX 6800 + 1x RX 6800 XT")
    print("=" * 60)
    
    # Run detection tests
    results = get_accurate_amd_gpu_count()
    
    # Analyze accuracy
    analysis = analyze_gpu_detection_accuracy(results)
    
    # Test integration
    integration_count = test_gpu_detector_integration()
    
    # Final summary
    print(f"\n🎉 Final Summary")
    print("=" * 30)
    print(f"Expected GPUs: 5 (4x RX 6800 + 1x RX 6800 XT)")
    print(f"Best Method: {analysis.get('best_method', 'Unknown')} ({analysis.get('best_accuracy', 0):.1f}% accuracy)")
    print(f"GPU Detector Integration: {integration_count} GPUs detected")
    
    # Save results
    final_results = {
        'expected_config': results['expected'],
        'detection_results': results,
        'accuracy_analysis': analysis,
        'integration_test': {
            'detected_count': integration_count,
            'success': integration_count >= 4  # At least 80% detection rate
        }
    }
    
    with open('accurate_gpu_detection_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"💾 Detailed results saved to: accurate_gpu_detection_results.json")
    
    # Recommendations
    if analysis.get('recommendations'):
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")

if __name__ == "__main__":
    main()
