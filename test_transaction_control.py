#!/usr/bin/env python3
"""
Test cluster manager with explicit transaction handling to debug hang
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

def test_cluster_manager_with_explicit_transactions():
    """Test with explicit transaction control"""
    print("🔍 Testing Cluster Manager with Explicit Transaction Control")
    print("⏰", datetime.now())
    print("========================================")
    
    config = {
        'CLUSTER_DB_HOST': 'localhost',
        'CLUSTER_DB_PORT': '5432',
        'CLUSTER_DB_NAME': 'bitinglip_cluster',
        'CLUSTER_DB_USER': 'cluster_manager',
        'CLUSTER_DB_PASSWORD': 'cluster_manager_2025!',
        'CLUSTER_DB_MIN_CONN': '1',
        'CLUSTER_DB_MAX_CONN': '5'  # Lower connection pool
    }
    
    try:
        # Initialize database
        database = ClusterDatabase(config)
        
        # Get or create a node
        query = "SELECT id FROM cluster_nodes ORDER BY last_heartbeat DESC LIMIT 1"
        nodes = database.execute_query(query, fetch=True)
        if nodes:
            node_id = nodes[0]['id']
            print(f"✅ Using existing node: {node_id}")
        else:
            print("❌ No nodes found")
            return False
        
        print("\n🔄 Starting monitoring loop simulation...")
        
        for i in range(5):
            print(f"\n--- Iteration {i+1} ---")
            
            def monitoring_iteration():
                try:
                    # Step 1: Heartbeat update with explicit connection handling
                    print("   1. Heartbeat update...")
                    start_time = time.time()
                    
                    # Use explicit connection control
                    with database.get_connection() as conn:
                        with conn.cursor() as cursor:
                            query = "UPDATE cluster_nodes SET last_heartbeat = NOW(), status = 'online' WHERE id = %s"
                            cursor.execute(query, (node_id,))
                            conn.commit()  # Explicit commit
                    
                    elapsed = time.time() - start_time
                    print(f"      ✅ Heartbeat updated ({elapsed:.3f}s)")
                    
                    # Step 2: System resource update
                    print("   2. System resource update...")
                    start_time = time.time()
                    
                    result = database.update_system_resources(
                        node_id=node_id,
                        cpu_percent=25.0 + i * 5,
                        memory_total=8*1024*1024*1024,
                        memory_used=3*1024*1024*1024,
                        disk_total=500*1024*1024*1024,
                        disk_used=100*1024*1024*1024,
                        network_sent=1000000,
                        network_recv=2000000,
                        load_avg=1.0
                    )
                    
                    elapsed = time.time() - start_time
                    print(f"      ✅ Resources updated ({elapsed:.3f}s) - result: {result}")
                    
                    # Step 3: Brief delay
                    print("   3. Sleeping...")
                    time.sleep(1)
                    
                    return True
                    
                except Exception as e:
                    print(f"      ❌ Iteration failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            # Run with timeout
            result_container = []
            def worker():
                result = monitoring_iteration()
                result_container.append(result)
            
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            
            # Wait with timeout
            thread.join(timeout=15)
            
            if thread.is_alive():
                print(f"❌ ITERATION {i+1} HUNG!")
                print("   Monitoring loop hanging - transaction or connection issue")
                return False
            else:
                if result_container and result_container[0]:
                    print(f"✅ Iteration {i+1} completed successfully")
                else:
                    print(f"❌ Iteration {i+1} failed")
                    return False
        
        print("\n========================================")
        print("✅ All monitoring iterations completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Test interrupted by user")
    sys.exit(0)

if __name__ == "__main__":
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    success = test_cluster_manager_with_explicit_transactions()
    
    if success:
        print("🎉 Transaction control test PASSED!")
    else:
        print("💥 Transaction control test FAILED!")
        sys.exit(1)
