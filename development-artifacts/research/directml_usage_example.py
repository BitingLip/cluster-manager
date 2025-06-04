"""
DirectML Usage Example for AMD RX 6800/6800 XT
System-wide installation approach
"""

import torch
import torch_directml

def test_directml_pytorch():
    """Test PyTorch with DirectML"""
    print("Testing PyTorch DirectML...")
    
    # Check available DirectML devices
    device_count = torch_directml.device_count()
    print(f"DirectML devices available: {device_count}")
    
    if device_count == 0:
        print("No DirectML devices found!")
        return False
    
    # Use DirectML device
    device = torch_directml.device()
    print(f"Using device: {device}")
    
    # Create tensors on DirectML device
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform computation
    import time
    start_time = time.time()
    z = torch.mm(x, y)
    end_time = time.time()
    
    print(f"Matrix multiplication completed in {end_time - start_time:.3f} seconds")
    print(f"Result shape: {z.shape}")
    
    return True

def test_onnx_directml():
    """Test ONNX Runtime with DirectML"""
    print("\nTesting ONNX Runtime DirectML...")
    
    try:
        import onnxruntime as ort
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            print("DirectML provider is available!")
            return True
        else:
            print("DirectML provider not found")
            return False
            
    except ImportError:
        print("ONNX Runtime not installed")
        return False

def main():
    """Run DirectML tests"""
    print("DirectML Test Suite for AMD RX 6800/6800 XT")
    print("=" * 50)
    
    pytorch_ok = test_directml_pytorch()
    onnx_ok = test_onnx_directml()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"PyTorch DirectML: {'✓' if pytorch_ok else '✗'}")
    print(f"ONNX DirectML: {'✓' if onnx_ok else '✗'}")
    
    if pytorch_ok or onnx_ok:
        print("\n✓ DirectML is working!")
        print("\nUsage tips:")
        print("- Always use system Python (not virtual environments)")
        print("- Use device = torch_directml.device() for PyTorch")
        print("- Use 'DmlExecutionProvider' for ONNX Runtime")
    else:
        print("\n✗ DirectML setup needs attention")
        print("- Check AMD driver installation")
        print("- Ensure running with system Python")
        print("- Verify packages installed system-wide")

if __name__ == "__main__":
    main()
