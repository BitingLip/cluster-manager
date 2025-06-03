#!/usr/bin/env python3
"""
Debug version of cluster manager startup with detailed logging
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(cluster_app_path))

# Load environment variables
env_file = Path(__file__).parent.parent / "config" / "cluster-manager-db.env"
if env_file.exists():
    import dotenv
    dotenv.load_dotenv(env_file)

from cluster_manager import ClusterManager
import time

def debug_cluster_startup():
    """Debug cluster manager startup with detailed step-by-step logging"""
    print("🔧 Debug Cluster Manager Startup...")
    print("=" * 60)
    
    try:
        print("Step 1: Creating ClusterManager instance...")
        cluster_manager = ClusterManager()
        print("✅ ClusterManager instance created")
        
        print("Step 2: Starting cluster manager...")
        start_time = time.time()
        success = cluster_manager.start()
        end_time = time.time()
        print(f"✅ Cluster manager start() completed in {end_time - start_time:.2f}s, success: {success}")
        
        if not success:
            print("❌ Cluster manager start() returned False")
            return False
        
        print("Step 3: Testing individual monitoring components...")
        
        # Test heartbeat update
        print("   3a: Testing heartbeat update...")
        start_time = time.time()
        heartbeat_success = cluster_manager.database.update_node_heartbeat(cluster_manager.node_id)
        end_time = time.time()
        print(f"   {'✅' if heartbeat_success else '❌'} Heartbeat update completed in {end_time - start_time:.2f}s")
        
        # Test system resources update 
        print("   3b: Testing system resources update...")
        start_time = time.time()
        try:
            cluster_manager._update_system_resources()
            end_time = time.time()
            print(f"   ✅ System resources update completed in {end_time - start_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            print(f"   ❌ System resources update failed in {end_time - start_time:.2f}s: {e}")
        
        # Test GPU resources update
        print("   3c: Testing GPU resources update...")
        start_time = time.time()
        try:
            cluster_manager._update_gpu_resources()
            end_time = time.time()
            print(f"   ✅ GPU resources update completed in {end_time - start_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            print(f"   ❌ GPU resources update failed in {end_time - start_time:.2f}s: {e}")
        
        # Test cleanup
        print("   3d: Testing stale nodes cleanup...")
        start_time = time.time()
        stale_count = cluster_manager.database.cleanup_stale_nodes(timeout_minutes=10)
        end_time = time.time()
        print(f"   ✅ Stale nodes cleanup completed in {end_time - start_time:.2f}s, cleaned: {stale_count}")
        
        print("\nStep 4: Running one complete monitoring iteration...")
        
        # Simulate one monitoring loop iteration
        start_time = time.time()
        
        print("   Iteration step 1: Heartbeat...")
        cluster_manager.database.update_node_heartbeat(cluster_manager.node_id)
        print("   ✅ Heartbeat updated")
        
        print("   Iteration step 2: System resources...")
        cluster_manager._update_system_resources()
        print("   ✅ System resources updated")
        
        print("   Iteration step 3: GPU resources...")
        cluster_manager._update_gpu_resources()
        print("   ✅ GPU resources updated")
        
        print("   Iteration step 4: Cleanup...")
        cluster_manager.database.cleanup_stale_nodes(timeout_minutes=10)
        print("   ✅ Cleanup completed")
        
        end_time = time.time()
        print(f"\n✅ Complete monitoring iteration took {end_time - start_time:.2f}s")
        
        print("\nStep 5: Closing connections...")
        cluster_manager.database.close()
        print("✅ Database connections closed")
        
        print("\n🎉 All debug tests passed! Cluster manager is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during debug startup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_cluster_startup()
    sys.exit(0 if success else 1)
