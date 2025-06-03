#!/usr/bin/env python3
"""
AMD DirectML GPU Detection Test
Specialized test for AMD Adrenalin 23 with DirectML support
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from gpu_detector import GPUDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_amd_directml_detection():
    """Test AMD GPU detection with DirectML support"""
    print("🚀 AMD DirectML GPU Detection Test")
    print("=" * 60)
    print("Testing for AMD Adrenalin 23 with DirectML support")
    print()
    
    detector = GPUDetector()
    
    try:
        # Test all GPU detection
        print("🔍 Phase 1: General GPU Detection")
        print("-" * 40)
        
        all_gpus = detector.detect_all_gpus()
        
        print(f"✅ Total GPUs detected: {len(all_gpus)}")
        
        if all_gpus:
            for i, gpu in enumerate(all_gpus):
                print(f"\n   GPU {i+1}:")
                print(f"      Name: {gpu.get('name', 'Unknown')}")
                print(f"      Vendor: {gpu.get('vendor', 'Unknown')}")
                print(f"      Memory: {gpu.get('memory_total_mb', 0):,} MB")
                print(f"      Driver Version: {gpu.get('driver_version', 'Unknown')}")
                
                # Check for DirectML indicators
                capabilities = gpu.get('ai_capabilities', {})
                if capabilities:
                    print(f"      AI Capabilities: {capabilities}")
                
                metadata = gpu.get('metadata', {})
                if metadata:
                    print(f"      Metadata: {metadata}")
        
        # Test specific AMD detection
        print(f"\n🔴 Phase 2: AMD-Specific Detection")
        print("-" * 40)
        
        amd_gpus = [gpu for gpu in all_gpus if gpu.get('vendor') == 'AMD']
        
        if amd_gpus:
            print(f"✅ AMD GPUs found: {len(amd_gpus)}")
            
            for i, gpu in enumerate(amd_gpus):
                print(f"\n   AMD GPU {i+1}:")
                print(f"      Model: {gpu.get('name', 'Unknown')}")
                print(f"      Architecture: {gpu.get('architecture', 'Unknown')}")
                print(f"      Compute Units: {gpu.get('compute_units', 'Unknown')}")
                print(f"      Memory: {gpu.get('memory_total_mb', 0):,} MB")
                
                # Test AI capability assessment
                if hasattr(detector, 'assess_ai_capability'):
                    ai_score = detector.assess_ai_capability(gpu)
                    print(f"      AI Capability Score: {ai_score:.2f}/10")
                
        else:
            print("⚠️ No AMD GPUs detected")
        
        # Test DirectML detection
        print(f"\n🤖 Phase 3: DirectML Support Assessment")
        print("-" * 40)
        
        directml_support = test_directml_support()
        
        if directml_support['available']:
            print("✅ DirectML is available!")
            print(f"   DirectML Version: {directml_support.get('version', 'Unknown')}")
            print(f"   Supported Devices: {directml_support.get('device_count', 0)}")
            
            if directml_support.get('devices'):
                for i, device in enumerate(directml_support['devices']):
                    print(f"   Device {i}: {device}")
        else:
            print("❌ DirectML not available")
            print(f"   Reason: {directml_support.get('error', 'Unknown')}")
        
        # Framework recommendations
        print(f"\n🎯 Phase 4: AI Framework Recommendations")
        print("-" * 40)
        
        recommendations = generate_amd_framework_recommendations(amd_gpus, directml_support)
        
        if recommendations:
            for category, rec in recommendations.items():
                print(f"\n   {category}:")
                print(f"      Recommendation: {rec['recommendation']}")
                print(f"      Confidence: {rec['confidence']}")
                if rec.get('notes'):
                    print(f"      Notes: {rec['notes']}")
                if rec.get('installation'):
                    print(f"      Installation: {rec['installation']}")
        
        # Performance assessment
        print(f"\n📊 Phase 5: Performance Assessment")
        print("-" * 40)
        
        performance = assess_amd_performance(amd_gpus, directml_support)
        
        print(f"   Overall Performance Score: {performance['overall_score']:.2f}/10")
        print(f"   DirectML Compatibility: {performance['directml_score']:.2f}/10")
        print(f"   Memory Score: {performance['memory_score']:.2f}/10")
        
        if performance.get('bottlenecks'):
            print(f"   Potential Bottlenecks: {', '.join(performance['bottlenecks'])}")
        
        if performance.get('optimizations'):
            print(f"\n   🎯 Optimization Suggestions:")
            for opt in performance['optimizations']:
                print(f"      • {opt}")
        
        # Save results
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'detected_gpus': all_gpus,
            'amd_gpus': amd_gpus,
            'directml_support': directml_support,
            'framework_recommendations': recommendations,
            'performance_assessment': performance,
            'test_status': 'success'
        }
        
        with open('amd_directml_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Test Complete!")
        print("=" * 60)
        print(f"✅ AMD GPU Detection: {'✅' if amd_gpus else '❌'}")
        print(f"✅ DirectML Support: {'✅' if directml_support['available'] else '❌'}")
        print(f"💾 Results saved to: amd_directml_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("AMD DirectML test failed")
        return False

def test_directml_support():
    """Test DirectML availability and device support"""
    try:
        # Try to import DirectML
        try:
            import tensorflow_directml
            directml_available = True
            version = getattr(tensorflow_directml, '__version__', 'Unknown')
        except ImportError:
            directml_available = False
            version = None
        
        if not directml_available:
            return {
                'available': False,
                'error': 'tensorflow-directml not installed'
            }
        
        # Try to detect DirectML devices
        try:
            import tensorflow as tf
            
            # List physical devices
            physical_devices = tf.config.list_physical_devices()
            gpu_devices = tf.config.list_physical_devices('GPU')
            
            device_info = []
            for device in gpu_devices:
                device_info.append(str(device))
            
            return {
                'available': True,
                'version': version,
                'device_count': len(gpu_devices),
                'devices': device_info,
                'all_devices': [str(d) for d in physical_devices]
            }
            
        except Exception as e:
            return {
                'available': True,
                'version': version,
                'device_count': 0,
                'error': f'Device detection failed: {e}'
            }
    
    except Exception as e:
        return {
            'available': False,
            'error': f'DirectML test failed: {e}'
        }

def generate_amd_framework_recommendations(amd_gpus, directml_support):
    """Generate AI framework recommendations for AMD GPUs"""
    if not amd_gpus:
        return {}
    
    recommendations = {}
    
    # DirectML + TensorFlow
    if directml_support['available']:
        recommendations['TensorFlow + DirectML'] = {
            'recommendation': 'Highly Recommended',
            'confidence': 'High',
            'notes': 'Native AMD GPU acceleration through DirectML',
            'installation': 'pip install tensorflow-directml'
        }
    else:
        recommendations['TensorFlow + DirectML'] = {
            'recommendation': 'Install Required',
            'confidence': 'High',
            'notes': 'Best option for AMD GPUs on Windows',
            'installation': 'pip install tensorflow-directml'
        }
    
    # PyTorch + DirectML
    recommendations['PyTorch + DirectML'] = {
        'recommendation': 'Recommended',
        'confidence': 'Medium',
        'notes': 'Experimental DirectML support for PyTorch',
        'installation': 'pip install torch-directml'
    }
    
    # ONNX Runtime + DirectML
    recommendations['ONNX Runtime + DirectML'] = {
        'recommendation': 'Recommended',
        'confidence': 'High',
        'notes': 'Excellent for model inference with AMD GPUs',
        'installation': 'pip install onnxruntime-directml'
    }
    
    # ROCm (Linux alternative)
    if sys.platform.startswith('linux'):
        recommendations['PyTorch + ROCm'] = {
            'recommendation': 'Linux Alternative',
            'confidence': 'Medium',
            'notes': 'Native AMD GPU support on Linux',
            'installation': 'Follow ROCm installation guide'
        }
    
    return recommendations

def assess_amd_performance(amd_gpus, directml_support):
    """Assess AMD GPU performance for AI workloads"""
    if not amd_gpus:
        return {
            'overall_score': 0,
            'directml_score': 0,
            'memory_score': 0,
            'bottlenecks': ['No AMD GPUs detected'],
            'optimizations': ['Install AMD GPU or drivers']
        }
    
    # Calculate scores
    total_memory = sum(gpu.get('memory_total_mb', 0) for gpu in amd_gpus) / 1024  # GB
    max_memory = max(gpu.get('memory_total_mb', 0) for gpu in amd_gpus) / 1024  # GB
    
    # Memory score (out of 10)
    memory_score = min(total_memory / 2, 10)  # 2GB per point, max 10
    
    # DirectML score
    directml_score = 8 if directml_support['available'] else 2
    
    # Overall score
    overall_score = (memory_score + directml_score) / 2
    
    # Identify bottlenecks and optimizations
    bottlenecks = []
    optimizations = []
    
    if total_memory < 8:
        bottlenecks.append('Limited GPU memory')
        optimizations.append('Consider upgrading to higher memory AMD GPU')
    
    if not directml_support['available']:
        bottlenecks.append('DirectML not installed')
        optimizations.append('Install tensorflow-directml for GPU acceleration')
    
    if len(amd_gpus) == 1:
        optimizations.append('Single GPU - consider multi-GPU setup for larger models')
    
    optimizations.append('Enable GPU memory growth in TensorFlow')
    optimizations.append('Use mixed precision training for better performance')
    
    return {
        'overall_score': overall_score,
        'directml_score': directml_score,
        'memory_score': memory_score,
        'bottlenecks': bottlenecks,
        'optimizations': optimizations,
        'amd_gpu_count': len(amd_gpus),
        'total_memory_gb': total_memory
    }

if __name__ == "__main__":
    success = test_amd_directml_detection()
    if success:
        print("\n🎉 AMD DirectML test completed successfully!")
    else:
        print("\n❌ AMD DirectML test failed!")
        sys.exit(1)
