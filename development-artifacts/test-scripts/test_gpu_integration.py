#!/usr/bin/env python3
"""
Test script to verify that all GPU-related modules work together
"""

import sys
import os

# Add the current directory to path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_gpu_detection():
    """Test basic GPU detection"""
    print("🔍 Testing GPU Detection")
    print("=" * 40)
    
    try:
        from gpu_detector import GPUDetector
        detector = GPUDetector()
        gpus = detector.detect_all_gpus()
        
        print(f"✅ GPU Detection successful!")
        print(f"Total GPUs found: {gpus['total_gpus']}")
        print(f"NVIDIA GPUs: {len(gpus['nvidia_gpus'])}")
        print(f"AMD GPUs: {len(gpus['amd_gpus'])}")
        
        return True
    except Exception as e:
        print(f"❌ GPU Detection failed: {e}")
        return False

def test_device_helpers():
    """Test device selection helpers"""
    print("\n🔧 Testing Device Selection Helpers")
    print("=" * 40)
    
    try:
        from gpu_device_helpers import MixedGPUDeviceSelector, auto_select_device
        
        # Test auto selection
        device = auto_select_device('auto')
        print(f"✅ Auto device selection successful: {device}")
        
        # Test advanced selector
        selector = MixedGPUDeviceSelector()
        print(f"Environment status: {selector.environment_status}")
        
        return True
    except Exception as e:
        print(f"❌ Device helpers failed: {e}")
        return False

def test_mixed_gpu_manager():
    """Test the main mixed GPU environment manager"""
    print("\n🚀 Testing Mixed GPU Environment Manager")
    print("=" * 40)
    
    try:
        from mixed_gpu_environment_manager import MixedGPUEnvironmentManager
        
        manager = MixedGPUEnvironmentManager()
        
        # Test GPU detection
        gpu_summary = manager.detect_all_gpus()
        print(f"✅ Manager GPU detection successful!")
        print(f"Mixed environment: {gpu_summary['mixed_environment']}")
        print(f"NVIDIA count: {gpu_summary['nvidia_count']}")
        print(f"AMD count: {gpu_summary['amd_count']}")
        
        # Test helper creation
        helper_file = manager.create_device_selection_helpers()
        if helper_file:
            print(f"✅ Device helpers available: {helper_file}")
        else:
            print("⚠️ Device helpers not available")
        
        return True
    except Exception as e:
        print(f"❌ Mixed GPU manager failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 GPU Environment Integration Test")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("GPU Detection", test_gpu_detection()))
    results.append(("Device Helpers", test_device_helpers()))
    results.append(("Mixed GPU Manager", test_mixed_gpu_manager()))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All import issues resolved! GPU environment is ready.")
    else:
        print("⚠️ Some issues remain. Check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
