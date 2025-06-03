#!/usr/bin/env python3
"""
Test system resource update in isolation to identify hang point
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import ClusterDatabase

def test_system_resources_with_timeout():
    """Test system resource update with timeout to prevent hanging"""
    print("🔍 Testing System Resource Update in Isolation")
    print("⏰", datetime.now())
    print("========================================")
    
    try:
        # Initialize database
        print("1. Initializing database connection...")
        config = {
            'CLUSTER_DB_HOST': 'localhost',
            'CLUSTER_DB_PORT': '5432',
            'CLUSTER_DB_NAME': 'bitinglip_cluster',
            'CLUSTER_DB_USER': 'cluster_manager',
            'CLUSTER_DB_PASSWORD': 'cluster_manager_2025!',
            'CLUSTER_DB_MIN_CONN': '1',
            'CLUSTER_DB_MAX_CONN': '3'
        }
        db = ClusterDatabase(config)
        print("✅ Database connected")
        
        # Get node info first
        print("2. Getting existing node info...")
        query = "SELECT id, node_type, status, last_heartbeat FROM cluster_nodes ORDER BY last_heartbeat DESC LIMIT 1"
        nodes = db.execute_query(query, fetch=True)
        
        if not nodes:
            print("❌ No nodes found in database")
            return
            
        node = nodes[0]
        node_id = node['id']
        print(f"✅ Found node: {node_id}")
        
        # Test system resource update with timeout
        print("3. Testing system resource update...")
        
        def update_system_resources():
            """Update system resources in separate thread"""
            try:
                print("   → Starting system resource update...")
                result = db.update_system_resources(
                    node_id=node_id,
                    cpu_percent=50.0,
                    memory_total=8*1024*1024*1024,  # 8GB
                    memory_used=4*1024*1024*1024,   # 4GB
                    disk_total=500*1024*1024*1024,  # 500GB
                    disk_used=200*1024*1024*1024,   # 200GB
                    network_sent=1000000,
                    network_recv=2000000,
                    load_avg=1.5
                )
                print(f"   → System resource update result: {result}")
                return result
            except Exception as e:
                print(f"   → System resource update failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Use threading to add timeout
        result_container = []
        def worker():
            result = update_system_resources()
            result_container.append(result)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        # Wait with timeout
        thread.join(timeout=15)  # 15 second timeout
        
        if thread.is_alive():
            print("❌ SYSTEM RESOURCE UPDATE HUNG! (timeout after 15 seconds)")
            print("   This confirms the issue is in the system resource update method")
            return False
        else:
            if result_container and result_container[0]:
                print("✅ System resource update completed successfully")
                
                # Verify the update
                print("4. Verifying system resource was updated...")
                verify_query = "SELECT * FROM system_resources WHERE node_id = %s ORDER BY timestamp DESC LIMIT 1"
                resources = db.execute_query(verify_query, (node_id,), fetch=True)
                if resources:
                    resource = resources[0]
                    print(f"✅ Updated resource timestamp: {resource['timestamp']}")
                    print(f"   CPU: {resource['cpu_percent']}%, Memory: {resource['memory_used']//1024//1024}MB")
                else:
                    print("❌ Could not verify system resource update")
            else:
                print("❌ System resource update returned False")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n========================================")
        print("✅ System resource isolation test completed!")

if __name__ == "__main__":
    test_system_resources_with_timeout()
