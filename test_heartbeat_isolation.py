#!/usr/bin/env python3
"""
Test heartbeat update in isolation to identify the exact hang point
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import ClusterDatabase

def test_heartbeat_with_timeout():
    """Test heartbeat update with timeout to prevent hanging"""
    print("🔍 Testing Heartbeat Update in Isolation")
    print("⏰", datetime.now())
    print("========================================")
    
    try:        # Initialize database
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
        print(f"✅ Found node: {node_id} (status: {node['status']}, last_heartbeat: {node['last_heartbeat']})")
        
        # Test heartbeat update with timeout
        print("3. Testing heartbeat update...")
        
        def update_heartbeat():
            """Update heartbeat in separate thread"""
            try:
                print("   → Starting heartbeat update...")
                result = db.update_node_heartbeat(node_id)
                print(f"   → Heartbeat update result: {result}")
                return result
            except Exception as e:
                print(f"   → Heartbeat update failed: {e}")
                return False
        
        # Use threading to add timeout
        result_container = []
        def worker():
            result = update_heartbeat()
            result_container.append(result)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        # Wait with timeout
        thread.join(timeout=10)  # 10 second timeout
        
        if thread.is_alive():
            print("❌ HEARTBEAT UPDATE HUNG! (timeout after 10 seconds)")
            print("   This confirms the issue is in the heartbeat update method")
            return False
        else:
            if result_container and result_container[0]:
                print("✅ Heartbeat update completed successfully")
                
                # Verify the update
                print("4. Verifying heartbeat was updated...")
                updated_nodes = db.execute_query(query, fetch=True)
                if updated_nodes:
                    updated_node = updated_nodes[0]
                    print(f"✅ Updated heartbeat: {updated_node['last_heartbeat']}")
                else:
                    print("❌ Could not verify heartbeat update")
            else:
                print("❌ Heartbeat update returned False")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n========================================")
        print("✅ Heartbeat isolation test completed!")

if __name__ == "__main__":
    test_heartbeat_with_timeout()
