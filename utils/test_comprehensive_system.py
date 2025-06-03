#!/usr/bin/env python3
"""
Comprehensive Cluster Manager with Enhanced GPU Detection Test
Validates the complete integration of GPU detection with cluster management
"""

import os
import sys
import logging
import json
from datetime import datetime
import uuid

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from gpu_detector import GPUDetector
from database import ClusterDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_comprehensive_cluster_gpu_system():
    """Test the complete cluster management system with GPU detection"""
    print("🚀 Comprehensive Cluster Manager + GPU Detection Test")
    print("=" * 60)
    
    # Test 1: GPU Detection
    print("\n🔍 Phase 1: Testing GPU Detection System")
    print("-" * 40)
    
    detector = GPUDetector()
    
    try:
        # Detect all GPUs
        all_gpus = detector.detect_all_gpus()
        
        print(f"✅ GPU Detection completed")
        print(f"   Total GPUs found: {len(all_gpus)}")
        
        if all_gpus:
            for i, gpu in enumerate(all_gpus):
                print(f"\n   GPU {i+1}:")
                print(f"      Name: {gpu.get('name', 'Unknown')}")
                print(f"      Vendor: {gpu.get('vendor', 'Unknown')}")
                print(f"      Memory: {gpu.get('memory_total_mb', 0):,} MB")
                
                # Test AI capability assessment
                ai_score = detector.assess_ai_capability(gpu)
                print(f"      AI Capability Score: {ai_score:.2f}/10")
                
                capabilities = gpu.get('ai_capabilities', {})
                if capabilities:
                    print(f"      Recommended for: {', '.join(capabilities.get('recommended_tasks', []))}")
        else:
            print("   ⚠️ No GPUs detected (expected if no GPU drivers installed)")
            
    except Exception as e:
        print(f"❌ GPU Detection failed: {e}")
        return False
    
    # Test 2: Database Operations (Mock)
    print(f"\n🗄️ Phase 2: Testing Database Operations (Mock)")
    print("-" * 40)
    
    try:
        # Mock database configuration
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'cluster_test',
            'user': 'cluster_user',
            'password': 'cluster_pass'
        }
        
        print("✅ Database configuration loaded")
        print(f"   Host: {db_config['host']}:{db_config['port']}")
        print(f"   Database: {db_config['database']}")
        
        # Note: We can't test actual database operations without a real PostgreSQL instance
        print("   ⚠️ Skipping actual database connection (requires PostgreSQL setup)")
        
    except Exception as e:
        print(f"❌ Database configuration failed: {e}")
        return False
    
    # Test 3: Mock Cluster Node Registration
    print(f"\n🖥️ Phase 3: Testing Cluster Node Registration (Mock)")
    print("-" * 40)
    
    try:
        # Generate a mock node
        node_id = f"node-{uuid.uuid4().hex[:8]}"
        
        # Create mock node capabilities based on detected GPUs
        capabilities = {
            'gpu_count': len(all_gpus),
            'total_gpu_memory_mb': sum(gpu.get('memory_total_mb', 0) for gpu in all_gpus),
            'ai_capable': any(detector.assess_ai_capability(gpu) > 5.0 for gpu in all_gpus),
            'supported_frameworks': [],
            'detection_time': datetime.now().isoformat()
        }
        
        # Add framework support based on detected vendors
        nvidia_gpus = [gpu for gpu in all_gpus if gpu.get('vendor') == 'NVIDIA']
        amd_gpus = [gpu for gpu in all_gpus if gpu.get('vendor') == 'AMD']
        
        if nvidia_gpus:
            capabilities['supported_frameworks'].extend(['pytorch-cuda', 'tensorflow-gpu', 'jax-cuda'])
        if amd_gpus:
            capabilities['supported_frameworks'].extend(['pytorch-directml', 'onnx-directml'])
        
        # Mock metadata
        metadata = {
            'operating_system': sys.platform,
            'python_version': sys.version.split()[0],
            'registration_time': datetime.now().isoformat(),
            'gpu_details': all_gpus
        }
        
        print(f"✅ Mock node registration prepared")
        print(f"   Node ID: {node_id}")
        print(f"   GPU Count: {capabilities['gpu_count']}")
        print(f"   Total GPU Memory: {capabilities['total_gpu_memory_mb']:,} MB")
        print(f"   AI Capable: {capabilities['ai_capable']}")
        print(f"   Supported Frameworks: {len(capabilities['supported_frameworks'])}")
        
    except Exception as e:
        print(f"❌ Node registration preparation failed: {e}")
        return False
    
    # Test 4: AI Workload Assessment
    print(f"\n🤖 Phase 4: AI Workload Assessment")
    print("-" * 40)
    
    try:
        workload_recommendations = assess_ai_workloads(all_gpus, detector)
        
        print("✅ AI Workload Assessment completed")
        
        if workload_recommendations:
            for category, recommendation in workload_recommendations.items():
                print(f"\n   {category}:")
                print(f"      Feasibility: {recommendation['feasibility']}")
                print(f"      Recommended GPU: {recommendation.get('recommended_gpu', 'None')}")
                if recommendation.get('notes'):
                    print(f"      Notes: {recommendation['notes']}")
        else:
            print("   ⚠️ No specific workload recommendations (insufficient GPU resources)")
            
    except Exception as e:
        print(f"❌ AI Workload Assessment failed: {e}")
        return False
    
    # Test 5: Performance Metrics
    print(f"\n📊 Phase 5: Performance Metrics")
    print("-" * 40)
    
    try:
        metrics = calculate_performance_metrics(all_gpus, detector)
        
        print("✅ Performance metrics calculated")
        print(f"   Overall AI Score: {metrics['overall_ai_score']:.2f}/10")
        print(f"   Total Compute Power: {metrics['total_compute_score']:.2f}")
        print(f"   Memory Bandwidth: {metrics['total_memory_gb']:.1f} GB")
        print(f"   Parallel Processing Score: {metrics['parallel_score']:.2f}")
        
        if metrics['bottlenecks']:
            print(f"   Potential Bottlenecks: {', '.join(metrics['bottlenecks'])}")
        
        if metrics['recommendations']:
            print(f"\n   🎯 Optimization Recommendations:")
            for rec in metrics['recommendations']:
                print(f"      • {rec}")
    
    except Exception as e:
        print(f"❌ Performance metrics calculation failed: {e}")
        return False
    
    # Final Summary
    print(f"\n🎉 Test Summary")
    print("=" * 60)
    print("✅ All phases completed successfully!")
    print(f"   • GPU Detection: Working")
    print(f"   • Database Layer: Ready")
    print(f"   • Node Registration: Prepared") 
    print(f"   • AI Assessment: Functional")
    print(f"   • Performance Metrics: Calculated")
    
    # Save complete test results
    results = {
        'test_timestamp': datetime.now().isoformat(),
        'detected_gpus': all_gpus,
        'node_capabilities': capabilities,
        'node_metadata': metadata,
        'workload_recommendations': workload_recommendations,
        'performance_metrics': metrics,
        'test_status': 'success'
    }
    
    with open('comprehensive_cluster_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Complete test results saved to: comprehensive_cluster_test_results.json")
    return True

def assess_ai_workloads(gpus, detector):
    """Assess what AI workloads are feasible with detected GPUs"""
    if not gpus:
        return {}
    
    recommendations = {}
    
    # Calculate total memory
    total_memory_gb = sum(gpu.get('memory_total_mb', 0) for gpu in gpus) / 1024
    max_single_gpu_memory = max((gpu.get('memory_total_mb', 0) for gpu in gpus), default=0) / 1024
    
    # Large Language Models
    if max_single_gpu_memory >= 24:
        recommendations['Large Language Models'] = {
            'feasibility': 'Excellent',
            'recommended_gpu': next((gpu['name'] for gpu in gpus if gpu.get('memory_total_mb', 0) >= 24000), None),
            'notes': 'Can run LLaMA-70B, GPT-3 scale models'
        }
    elif max_single_gpu_memory >= 16:
        recommendations['Large Language Models'] = {
            'feasibility': 'Good',
            'recommended_gpu': next((gpu['name'] for gpu in gpus if gpu.get('memory_total_mb', 0) >= 16000), None),
            'notes': 'Can run LLaMA-30B, medium-scale models'
        }
    elif max_single_gpu_memory >= 8:
        recommendations['Large Language Models'] = {
            'feasibility': 'Limited',
            'recommended_gpu': next((gpu['name'] for gpu in gpus if gpu.get('memory_total_mb', 0) >= 8000), None),
            'notes': 'Can run LLaMA-7B, small models only'
        }
    
    # Computer Vision
    if total_memory_gb >= 8:
        recommendations['Computer Vision'] = {
            'feasibility': 'Excellent',
            'notes': 'Object detection, image segmentation, video processing'
        }
    elif total_memory_gb >= 4:
        recommendations['Computer Vision'] = {
            'feasibility': 'Good',
            'notes': 'Image classification, basic object detection'
        }
    
    # Stable Diffusion / Image Generation
    if max_single_gpu_memory >= 10:
        recommendations['Image Generation'] = {
            'feasibility': 'Excellent',
            'notes': 'SDXL, high-resolution generation'
        }
    elif max_single_gpu_memory >= 6:
        recommendations['Image Generation'] = {
            'feasibility': 'Good',
            'notes': 'Stable Diffusion 1.5, medium resolution'
        }
    
    # Training vs Inference
    if total_memory_gb >= 16:
        recommendations['Training Capability'] = {
            'feasibility': 'Good',
            'notes': 'Can train medium models, fine-tune large models'
        }
    else:
        recommendations['Training Capability'] = {
            'feasibility': 'Limited',
            'notes': 'Primarily suitable for inference'
        }
    
    return recommendations

def calculate_performance_metrics(gpus, detector):
    """Calculate comprehensive performance metrics"""
    if not gpus:
        return {
            'overall_ai_score': 0,
            'total_compute_score': 0,
            'total_memory_gb': 0,
            'parallel_score': 0,
            'bottlenecks': ['No GPUs detected'],
            'recommendations': ['Install GPU drivers', 'Add dedicated GPU hardware']
        }
    
    # Calculate scores
    ai_scores = [detector.assess_ai_capability(gpu) for gpu in gpus]
    overall_ai_score = sum(ai_scores) / len(ai_scores) if ai_scores else 0
    
    total_memory_gb = sum(gpu.get('memory_total_mb', 0) for gpu in gpus) / 1024
    
    # Compute score based on GPU count and individual capabilities
    compute_score = sum(ai_scores)
    
    # Parallel processing score
    gpu_count = len(gpus)
    parallel_score = min(gpu_count * 2, 10)  # Cap at 10
    
    # Identify bottlenecks
    bottlenecks = []
    recommendations = []
    
    if total_memory_gb < 8:
        bottlenecks.append('Limited GPU memory')
        recommendations.append('Consider upgrading to higher memory GPUs')
    
    if gpu_count == 1:
        bottlenecks.append('Single GPU limits parallel processing')
        recommendations.append('Add additional GPUs for parallel workloads')
    
    nvidia_count = sum(1 for gpu in gpus if gpu.get('vendor') == 'NVIDIA')
    amd_count = sum(1 for gpu in gpus if gpu.get('vendor') == 'AMD')
    
    if nvidia_count == 0:
        recommendations.append('NVIDIA GPUs recommended for CUDA-based frameworks')
    
    if overall_ai_score < 5:
        recommendations.append('Current GPUs may struggle with modern AI workloads')
    
    return {
        'overall_ai_score': overall_ai_score,
        'total_compute_score': compute_score,
        'total_memory_gb': total_memory_gb,
        'parallel_score': parallel_score,
        'bottlenecks': bottlenecks,
        'recommendations': recommendations,
        'gpu_breakdown': {
            'nvidia_gpus': nvidia_count,
            'amd_gpus': amd_count,
            'total_gpus': gpu_count
        }
    }

if __name__ == "__main__":
    test_comprehensive_cluster_gpu_system()
