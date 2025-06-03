#!/usr/bin/env python3
"""
GPU Detection Test Utility
Tests both NVIDIA and AMD GPU detection with AI capability assessment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cluster_manager import ClusterManager
import json
import logging

def test_gpu_detection():
    """Test GPU detection capabilities"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Testing Enhanced GPU Detection for AI Workloads")
    print("=" * 60)
    
    try:
        # Create cluster manager instance (without starting full services)
        manager = ClusterManager()
        
        # Test NVIDIA GPU detection
        print("\n🟢 Testing NVIDIA GPU Detection...")
        nvidia_gpus = manager._detect_nvidia_gpus()
        
        if nvidia_gpus:
            print(f"   ✅ Found {len(nvidia_gpus)} NVIDIA GPU(s)")
            for gpu in nvidia_gpus:
                print(f"   📊 {gpu['name']}")
                print(f"      💾 VRAM: {gpu['memory_total_mb']:,} MB")
                print(f"      🏗️ Architecture: {gpu['metadata']['architecture']}")
                print(f"      🧠 AI Capabilities:")
                ai_caps = gpu['ai_capabilities']
                print(f"         • Tensor Cores: {ai_caps.get('tensor_cores', False)}")
                print(f"         • FP16 Support: {ai_caps.get('fp16_support', False)}")
                print(f"         • Training Capable: {ai_caps.get('suitable_for_training', False)}")
                print(f"         • Inference Capable: {ai_caps.get('suitable_for_inference', False)}")
                if ai_caps.get('recommended_models'):
                    print(f"         • Recommended Models: {', '.join(ai_caps['recommended_models'][:3])}")
                print()
        else:
            print("   ⚪ No NVIDIA GPUs detected")
        
        # Test AMD GPU detection
        print("\n🔴 Testing AMD GPU Detection...")
        amd_gpus = manager._detect_amd_gpus()
        
        if amd_gpus:
            print(f"   ✅ Found {len(amd_gpus)} AMD GPU(s)")
            for gpu in amd_gpus:
                print(f"   📊 {gpu['name']}")
                print(f"      💾 VRAM: {gpu['memory_total_mb']:,} MB")
                print(f"      🏗️ Architecture: {gpu['metadata']['architecture']}")
                print(f"      🧠 AI Capabilities:")
                ai_caps = gpu['ai_capabilities']
                print(f"         • DirectML Support: {ai_caps.get('directml_support', False)}")
                print(f"         • FP16 Support: {ai_caps.get('fp16_support', False)}")
                print(f"         • Quantization Support: {ai_caps.get('quantization_support', False)}")
                print(f"         • Driver Optimized: {ai_caps.get('driver_optimized', False)}")
                print(f"         • Training Capable: {ai_caps.get('suitable_for_training', False)}")
                print(f"         • Inference Capable: {ai_caps.get('suitable_for_inference', False)}")
                if ai_caps.get('recommended_models'):
                    print(f"         • Recommended Models: {', '.join(ai_caps['recommended_models'][:3])}")
                print(f"         • Frameworks: {', '.join(gpu['framework_support'][:4])}")
                if ai_caps.get('architecture_notes'):
                    print(f"         • Notes: {ai_caps['architecture_notes']}")
                print()
        else:
            print("   ⚪ No AMD GPUs detected")
        
        # Combined summary
        total_gpus = len(nvidia_gpus) + len(amd_gpus)
        print(f"\n📈 GPU Detection Summary")
        print(f"   • Total GPUs: {total_gpus}")
        print(f"   • NVIDIA GPUs: {len(nvidia_gpus)}")
        print(f"   • AMD GPUs: {len(amd_gpus)}")
        
        if total_gpus > 0:
            print(f"\n🚀 AI Workload Optimization Recommendations:")
            
            # Calculate total VRAM
            total_vram_gb = sum(gpu['memory_total_mb'] for gpu in nvidia_gpus + amd_gpus) / 1024
            print(f"   • Total VRAM Available: {total_vram_gb:.1f} GB")
            
            # Count by capability
            training_gpus = sum(1 for gpu in nvidia_gpus + amd_gpus 
                              if gpu['ai_capabilities'].get('suitable_for_training', False))
            inference_gpus = sum(1 for gpu in nvidia_gpus + amd_gpus 
                               if gpu['ai_capabilities'].get('suitable_for_inference', False))
            
            print(f"   • Training-Capable GPUs: {training_gpus}")
            print(f"   • Inference-Capable GPUs: {inference_gpus}")
            
            # Framework recommendations
            frameworks = set()
            for gpu in nvidia_gpus + amd_gpus:
                frameworks.update(gpu.get('framework_support', []))
            
            print(f"   • Supported Frameworks: {', '.join(sorted(frameworks))}")
            
            # Model recommendations based on available VRAM
            if total_vram_gb >= 40:
                print(f"   • Can run: LLaMA-65B, GPT-3.5 class models, Large-scale training")
            elif total_vram_gb >= 20:
                print(f"   • Can run: LLaMA-30B, Stable Diffusion XL, Medium-scale training")
            elif total_vram_gb >= 12:
                print(f"   • Can run: LLaMA-13B, Stable Diffusion 2.1, Small-scale training")
            elif total_vram_gb >= 6:
                print(f"   • Can run: LLaMA-7B, Stable Diffusion 1.5, Inference workloads")
            else:
                print(f"   • Can run: Small models, Basic CNNs, Educational projects")
            
            # AMD-specific recommendations
            if amd_gpus:
                print(f"\n🔴 AMD-Specific Recommendations:")
                print(f"   • Install AMD Adrenalin Edition 23.40.27.06+ for AI optimizations")
                print(f"   • Use PyTorch-DirectML for training: pip install pytorch-directml")
                print(f"   • Use ONNX Runtime with DirectML for optimized inference")
                print(f"   • Consider Microsoft Olive for model optimization")
                
                rdna3_count = sum(1 for gpu in amd_gpus if 'RDNA3' in gpu['metadata']['architecture'])
                if rdna3_count > 0:
                    print(f"   • RDNA3 GPUs detected: Use latest drivers for 2x performance boost")
        
        # Save detailed results
        results = {
            'nvidia_gpus': nvidia_gpus,
            'amd_gpus': amd_gpus,
            'summary': {
                'total_gpus': total_gpus,
                'nvidia_count': len(nvidia_gpus),
                'amd_count': len(amd_gpus),
                'total_vram_gb': total_vram_gb if total_gpus > 0 else 0,
                'training_capable': training_gpus if total_gpus > 0 else 0,
                'inference_capable': inference_gpus if total_gpus > 0 else 0
            }
        }
        
        with open('gpu_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: gpu_detection_results.json")
        
    except Exception as e:
        print(f"❌ Error during GPU detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_detection()
