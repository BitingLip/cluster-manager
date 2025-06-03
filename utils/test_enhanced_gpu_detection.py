#!/usr/bin/env python3
"""
Enhanced GPU Detection Test for Cluster Manager
Tests both NVIDIA and AMD GPU detection with AI optimization analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gpu_detector import GPUDetector
import json
import logging

def test_enhanced_gpu_detection():
    """Test the enhanced GPU detection system"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Enhanced GPU Detection for AI Workloads")
    print("=" * 60)
    
    try:
        # Create GPU detector
        detector = GPUDetector()
        
        print("\n🔍 Detecting All Available GPUs...")
        all_gpus = detector.detect_all_gpus()
        
        if not all_gpus:
            print("   ⚪ No GPUs detected on this system")
            return
        
        print(f"\n✅ Found {len(all_gpus)} GPU(s) Total")
        print("=" * 60)
        
        # Detailed GPU analysis
        nvidia_count = 0
        amd_count = 0
        total_vram_gb = 0
        
        for i, gpu in enumerate(all_gpus):
            print(f"\n📊 GPU {i+1}: {gpu['name']}")
            print(f"   🏢 Vendor: {gpu['vendor']}")
            print(f"   💾 VRAM: {gpu['memory_total_mb']:,} MB ({gpu['memory_total_mb']/1024:.1f} GB)")
            
            if gpu['vendor'] == 'NVIDIA':
                nvidia_count += 1
                print(f"   🧮 Compute Capability: {gpu.get('compute_capability', 'Unknown')}")
                print(f"   🌡️ Temperature: {gpu.get('temperature_celsius', 'N/A')}°C")
                print(f"   ⚡ Power Usage: {gpu.get('power_usage_watts', 'N/A')} W")
            elif gpu['vendor'] == 'AMD':
                amd_count += 1
                if 'compute_units' in gpu:
                    print(f"   🧮 Compute Units: {gpu['compute_units']}")
            
            print(f"   🏗️ Architecture: {gpu['metadata']['architecture']}")
            print(f"   🚗 Driver: {gpu.get('driver_version', 'Unknown')}")
            print(f"   📈 Utilization: {gpu.get('utilization_percent', 0)}%")
            
            # AI Capabilities Analysis
            ai_caps = gpu['ai_capabilities']
            print(f"\n   🧠 AI Capabilities:")
            
            if gpu['vendor'] == 'NVIDIA':
                print(f"      • Tensor Cores: {'✅' if ai_caps.get('tensor_cores') else '❌'}")
                print(f"      • FP16 Support: {'✅' if ai_caps.get('fp16_support') else '❌'}")
                print(f"      • INT8 Support: {'✅' if ai_caps.get('int8_support') else '❌'}")
            else:  # AMD
                print(f"      • DirectML Support: {'✅' if ai_caps.get('directml_support') else '❌'}")
                print(f"      • FP16 Support: {'✅' if ai_caps.get('fp16_support') else '❌'}")
                print(f"      • Quantization Support: {'✅' if ai_caps.get('quantization_support') else '❌'}")
                print(f"      • Driver Optimized: {'✅' if ai_caps.get('driver_optimized') else '❌'}")
                if ai_caps.get('ai_accelerators'):
                    print(f"      • AI Accelerators: ✅")
            
            print(f"      • Performance Tier: {ai_caps.get('performance_tier', 'unknown').title()}")
            print(f"      • Training Capable: {'✅' if ai_caps.get('suitable_for_training') else '❌'}")
            print(f"      • Inference Capable: {'✅' if ai_caps.get('suitable_for_inference') else '✅'}")
            
            # Recommended Models
            models = ai_caps.get('recommended_models', [])
            if models:
                print(f"      • Recommended Models:")
                for model in models[:4]:  # Show top 4 models
                    print(f"        - {model}")
                if len(models) > 4:
                    print(f"        - ... and {len(models)-4} more")
            
            # Framework Support
            frameworks = gpu.get('framework_support', [])
            print(f"      • Supported Frameworks: {', '.join(frameworks[:4])}")
            if len(frameworks) > 4:
                print(f"        + {len(frameworks)-4} more...")
            
            # Architecture Notes (AMD specific)
            if gpu['vendor'] == 'AMD' and ai_caps.get('architecture_notes'):
                print(f"      • Notes: {ai_caps['architecture_notes']}")
            
            total_vram_gb += gpu['memory_total_mb'] / 1024
        
        # Overall System Analysis
        print(f"\n🖥️ System GPU Summary")
        print("=" * 60)
        print(f"   • Total GPUs: {len(all_gpus)}")
        print(f"   • NVIDIA GPUs: {nvidia_count}")
        print(f"   • AMD GPUs: {amd_count}")
        print(f"   • Total VRAM: {total_vram_gb:.1f} GB")
        
        # AI Workload Recommendations
        print(f"\n🤖 AI Workload Recommendations")
        print("=" * 60)
        
        # Analyze overall capabilities
        training_gpus = sum(1 for gpu in all_gpus 
                          if gpu['ai_capabilities'].get('suitable_for_training', False))
        inference_gpus = sum(1 for gpu in all_gpus 
                           if gpu['ai_capabilities'].get('suitable_for_inference', False))
        
        print(f"   • Training-Capable GPUs: {training_gpus}")
        print(f"   • Inference-Capable GPUs: {inference_gpus}")
        
        # Model recommendations based on total VRAM
        print(f"\n   🎯 Model Size Recommendations:")
        if total_vram_gb >= 40:
            print(f"      ✅ Can run LLaMA-65B (4-bit) across multiple GPUs")
            print(f"      ✅ Large-scale model training")
            print(f"      ✅ Multiple concurrent AI workloads")
        elif total_vram_gb >= 24:
            print(f"      ✅ Can run LLaMA-30B (4-bit) on single GPU")
            print(f"      ✅ Stable Diffusion XL with high quality")
            print(f"      ✅ Medium-scale model training")
        elif total_vram_gb >= 16:
            print(f"      ✅ Can run LLaMA-13B (4-bit)")
            print(f"      ✅ Stable Diffusion 2.1")
            print(f"      ✅ Small-scale model training and fine-tuning")
        elif total_vram_gb >= 8:
            print(f"      ✅ Can run LLaMA-7B (4-bit)")
            print(f"      ✅ Stable Diffusion 1.5")
            print(f"      ✅ Small model inference")
        elif total_vram_gb >= 4:
            print(f"      ✅ Can run Phi-3 mini")
            print(f"      ✅ Basic CNNs and small transformers")
        else:
            print(f"      ⚠️ Limited AI capabilities with current hardware")
        
        # Multi-GPU considerations
        if len(all_gpus) > 1:
            print(f"\n   🔗 Multi-GPU Considerations:")
            print(f"      • {len(all_gpus)} GPUs available for parallel workloads")
            
            if nvidia_count > 1:
                print(f"      • NVIDIA multi-GPU: CUDA/NCCL supported for distributed training")
            
            if amd_count > 1:
                print(f"      • AMD multi-GPU: Limited Windows support, consider parallel inference")
                print(f"      • Alternative: Use each GPU for independent tasks")
        
        # Framework-specific recommendations
        print(f"\n   🛠️ Framework Setup Recommendations:")
        
        if nvidia_count > 0:
            print(f"      🟢 NVIDIA GPUs detected:")
            print(f"         • Install PyTorch with CUDA support")
            print(f"         • Use TensorFlow-GPU")
            print(f"         • Consider Triton Inference Server for production")
        
        if amd_count > 0:
            print(f"      🔴 AMD GPUs detected:")
            rdna_gpus = sum(1 for gpu in all_gpus 
                          if gpu['vendor'] == 'AMD' and 'RDNA' in gpu['metadata']['architecture'])
            
            if rdna_gpus > 0:
                print(f"         • Install AMD Adrenalin Edition 23.40.27.06+ driver")
                print(f"         • Use PyTorch-DirectML: pip install pytorch-directml")
                print(f"         • Use ONNX Runtime with DirectML provider")
                print(f"         • Consider Microsoft Olive for model optimization")
            else:
                print(f"         • Legacy AMD GPU detected - basic DirectML support")
                print(f"         • Use OpenCL backends where available")
        
        # Performance optimization tips
        print(f"\n   ⚡ Performance Optimization Tips:")
        for gpu in all_gpus:
            if gpu['vendor'] == 'AMD' and 'RDNA3' in gpu['metadata']['architecture']:
                print(f"      • {gpu['name']}: Update to latest drivers for 2x AI performance boost")
            elif gpu['vendor'] == 'NVIDIA' and gpu['ai_capabilities'].get('tensor_cores'):
                print(f"      • {gpu['name']}: Enable mixed precision (FP16) for faster training")
        
        # Save detailed results
        results = {
            'detected_gpus': all_gpus,
            'summary': {
                'total_gpus': len(all_gpus),
                'nvidia_count': nvidia_count,
                'amd_count': amd_count,
                'total_vram_gb': total_vram_gb,
                'training_capable': training_gpus,
                'inference_capable': inference_gpus
            },
            'timestamp': detector.detect_all_gpus()[0]['metadata']['collection_time'] if all_gpus else None
        }
        
        with open('enhanced_gpu_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: enhanced_gpu_detection_results.json")
        
    except Exception as e:
        print(f"❌ Error during GPU detection: {e}")
        import traceback
        traceback.print_exc()

def analyze_mining_rig_potential():
    """Analyze potential for repurposing mining rigs for AI"""
    print(f"\n⛏️ Mining Rig to AI Workstation Analysis")
    print("=" * 60)
    
    detector = GPUDetector()
    all_gpus = detector.detect_all_gpus()
    
    amd_gpus = [gpu for gpu in all_gpus if gpu['vendor'] == 'AMD']
    
    if not amd_gpus:
        print("   ⚪ No AMD GPUs detected (common in mining setups)")
        return
    
    print(f"   📊 Found {len(amd_gpus)} AMD GPU(s) - analyzing mining rig potential...")
    
    mining_suitable = []
    ai_suitable = []
    
    for gpu in amd_gpus:
        name = gpu['name']
        memory_gb = gpu['memory_total_mb'] / 1024
        arch = gpu['metadata']['architecture']
        
        # Common mining GPUs
        mining_cards = ['RX 580', 'RX 570', 'RX 480', 'RX 470', 'RX 5700', 'RX 6600', 'RX 6700', 'RX 6800', 'RX 6900', 'Vega']
        
        is_mining_card = any(card in name for card in mining_cards)
        if is_mining_card:
            mining_suitable.append(gpu)
        
        ai_caps = gpu['ai_capabilities']
        if ai_caps.get('suitable_for_inference') or ai_caps.get('suitable_for_training'):
            ai_suitable.append(gpu)
    
    if mining_suitable:
        print(f"\n   ⛏️ Detected Mining-Era GPUs: {len(mining_suitable)}")
        for gpu in mining_suitable:
            name = gpu['name']
            memory_gb = gpu['memory_total_mb'] / 1024
            ai_caps = gpu['ai_capabilities']
            
            print(f"      • {name} ({memory_gb:.0f}GB)")
            print(f"        - Architecture: {gpu['metadata']['architecture']}")
            print(f"        - AI Tier: {ai_caps.get('performance_tier', 'unknown').title()}")
            
            models = ai_caps.get('recommended_models', [])
            if models:
                print(f"        - Can run: {', '.join(models[:2])}")
    
    if ai_suitable:
        total_ai_vram = sum(gpu['memory_total_mb'] for gpu in ai_suitable) / 1024
        print(f"\n   🤖 AI-Suitable GPUs: {len(ai_suitable)} ({total_ai_vram:.0f}GB total)")
        print(f"   💡 Recommendation: Repurpose for AI workloads!")
        
        if len(ai_suitable) > 1:
            print(f"   🔗 Multi-GPU setup: Run {len(ai_suitable)} parallel inference tasks")
    
    else:
        print(f"\n   ⚠️ Limited AI potential with current mining GPUs")
        print(f"   💡 Consider upgrading to RDNA2+ for better AI performance")

if __name__ == "__main__":
    test_enhanced_gpu_detection()
    analyze_mining_rig_potential()
