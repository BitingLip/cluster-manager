#!/usr/bin/env python3
"""
Test complete monitoring loop sequence in isolation to identify hang point
"""

import sys
import os
import time
import threading
import signal
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import ClusterDatabase

class MonitoringLoopTest:
    def __init__(self):
        self.config = {
            'CLUSTER_DB_HOST': 'localhost',
            'CLUSTER_DB_PORT': '5432',
            'CLUSTER_DB_NAME': 'bitinglip_cluster',
            'CLUSTER_DB_USER': 'cluster_manager',
            'CLUSTER_DB_PASSWORD': 'cluster_manager_2025!',
            'CLUSTER_DB_MIN_CONN': '1',
            'CLUSTER_DB_MAX_CONN': '3'
        }
        self.database = None
        self.node_id = None
        self.shutdown = False

    def setup(self):
        """Initialize test setup"""
        print("🔧 Setting up monitoring loop test...")
        
        # Initialize database
        self.database = ClusterDatabase(self.config)
        
        # Get existing node
        query = "SELECT id FROM cluster_nodes ORDER BY last_heartbeat DESC LIMIT 1"
        nodes = self.database.execute_query(query, fetch=True)
        if nodes:
            self.node_id = nodes[0]['id']
            print(f"✅ Using existing node: {self.node_id}")
        else:
            print("❌ No nodes found")
            return False
        return True

    def single_monitoring_iteration(self):
        """Execute one complete monitoring iteration"""
        step = 1
        try:
            print(f"   Step {step}: Updating node heartbeat...")
            step += 1
            start_time = time.time()
            self.database.update_node_heartbeat(self.node_id)
            elapsed = time.time() - start_time
            print(f"      ✅ Heartbeat updated ({elapsed:.2f}s)")
            
            print(f"   Step {step}: Updating system resources...")
            step += 1
            start_time = time.time()
            self.database.update_system_resources(
                node_id=self.node_id,
                cpu_percent=45.0,
                memory_total=8*1024*1024*1024,
                memory_used=3*1024*1024*1024,
                disk_total=500*1024*1024*1024,
                disk_used=150*1024*1024*1024,
                network_sent=500000,
                network_recv=1000000,
                load_avg=1.2
            )
            elapsed = time.time() - start_time
            print(f"      ✅ System resources updated ({elapsed:.2f}s)")
            
            print(f"   Step {step}: Updating GPU resources...")
            step += 1
            start_time = time.time()
            # Just update with empty GPU data (no GPUs)
            self.database.update_gpu_resources(self.node_id, [])
            elapsed = time.time() - start_time
            print(f"      ✅ GPU resources updated ({elapsed:.2f}s)")
            
            print(f"   Step {step}: Cleaning up stale nodes...")
            step += 1
            start_time = time.time()
            stale_count = self.database.cleanup_stale_nodes(timeout_minutes=10)
            elapsed = time.time() - start_time
            print(f"      ✅ Stale cleanup completed ({elapsed:.2f}s) - {stale_count} nodes cleaned")
            
            return True
            
        except Exception as e:
            print(f"      ❌ Failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_monitoring_loop(self, iterations=3):
        """Test multiple monitoring loop iterations"""
        print("🔍 Testing Complete Monitoring Loop Sequence")
        print("⏰", datetime.now())
        print("========================================")
        
        if not self.setup():
            return False
        
        # Test multiple iterations
        for i in range(iterations):
            print(f"\n🔄 Iteration {i+1}/{iterations}")
            
            def monitoring_worker():
                return self.single_monitoring_iteration()
            
            # Use threading with timeout
            result_container = []
            def worker():
                result = monitoring_worker()
                result_container.append(result)
            
            thread = threading.Thread(target=worker)
            thread.daemon = True
            start_time = time.time()
            thread.start()
            
            # Wait with timeout
            thread.join(timeout=30)  # 30 second timeout per iteration
            
            if thread.is_alive():
                elapsed = time.time() - start_time
                print(f"❌ MONITORING ITERATION {i+1} HUNG! (timeout after {elapsed:.1f}s)")
                print("   The monitoring loop is hanging somewhere in the sequence")
                return False
            else:
                if result_container and result_container[0]:
                    elapsed = time.time() - start_time
                    print(f"✅ Iteration {i+1} completed successfully ({elapsed:.2f}s)")
                else:
                    print(f"❌ Iteration {i+1} failed")
                    return False
            
            # Brief sleep between iterations
            if i < iterations - 1:
                print("   Sleeping 2 seconds before next iteration...")
                time.sleep(2)
        
        print("\n========================================")
        print("✅ All monitoring loop iterations completed successfully!")
        return True

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Test interrupted by user")
    sys.exit(0)

if __name__ == "__main__":
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    test = MonitoringLoopTest()
    success = test.test_monitoring_loop()
    
    if success:
        print("🎉 Monitoring loop test PASSED - no hanging detected!")
    else:
        print("💥 Monitoring loop test FAILED - hanging detected!")
        sys.exit(1)
