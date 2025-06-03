#!/usr/bin/env python3
"""
Test the exact _update_system_resources method from cluster manager
"""

import sys
import os
import time
import threading
import psutil
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import ClusterDatabase

def test_real_system_resources_update():
    """Test the actual system resource collection logic"""
    print("🔍 Testing Real System Resources Collection")
    print("⏰", datetime.now())
    print("========================================")
    
    try:
        # Initialize database
        config = {
            'CLUSTER_DB_HOST': 'localhost',
            'CLUSTER_DB_PORT': '5432',
            'CLUSTER_DB_NAME': 'bitinglip_cluster',
            'CLUSTER_DB_USER': 'cluster_manager',
            'CLUSTER_DB_PASSWORD': 'cluster_manager_2025!',
            'CLUSTER_DB_MIN_CONN': '1',
            'CLUSTER_DB_MAX_CONN': '3'
        }
        database = ClusterDatabase(config)
        
        # Get node
        query = "SELECT id FROM cluster_nodes ORDER BY last_heartbeat DESC LIMIT 1"
        nodes = database.execute_query(query, fetch=True)
        if not nodes:
            print("❌ No nodes found")
            return
        node_id = nodes[0]['id']
        
        def update_system_resources_real():
            """This is the exact code from cluster_manager.py _update_system_resources"""
            try:
                print("   1. Getting CPU information...")
                # Get CPU information
                cpu_count = psutil.cpu_count()
                cpu_usage = psutil.cpu_percent(interval=1)  # This takes 1 second!
                print(f"      ✅ CPU: {cpu_count} cores, {cpu_usage}% usage")
                
                print("   2. Getting memory information...")
                # Get memory information
                memory = psutil.virtual_memory()
                memory_total_mb = memory.total // (1024 * 1024)
                memory_used_mb = memory.used // (1024 * 1024)
                memory_available_mb = memory.available // (1024 * 1024)
                print(f"      ✅ Memory: {memory_total_mb}MB total, {memory_used_mb}MB used")
                
                print("   3. Getting disk information...")
                # Get disk information
                try:
                    # Use different disk paths for different platforms
                    if sys.platform == 'win32':
                        disk = psutil.disk_usage('C:')
                    else:
                        disk = psutil.disk_usage('/')
                    disk_total_gb = disk.total // (1024 * 1024 * 1024)
                    disk_used_gb = disk.used // (1024 * 1024 * 1024)
                    disk_available_gb = disk.free // (1024 * 1024 * 1024)
                    print(f"      ✅ Disk: {disk_total_gb}GB total, {disk_used_gb}GB used")
                except Exception as e:
                    print(f"      ⚠️ Disk warning: {e}")
                    disk_total_gb = 0
                    disk_used_gb = 0
                    disk_available_gb = 0
                
                print("   4. Getting network information...")
                # Get network information
                net_io = psutil.net_io_counters()
                network_rx_bytes = net_io.bytes_recv
                network_tx_bytes = net_io.bytes_sent
                print(f"      ✅ Network: {network_rx_bytes} RX, {network_tx_bytes} TX")
                
                print("   5. Getting load average...")
                # Get load average (Unix only)
                try:
                    load_avg = os.getloadavg()
                    load_average_1m = load_avg[0]
                    load_average_5m = load_avg[1]
                    load_average_15m = load_avg[2]
                    print(f"      ✅ Load: {load_average_1m:.2f}, {load_average_5m:.2f}, {load_average_15m:.2f}")
                except (AttributeError, OSError):
                    # Windows doesn't have load average
                    load_average_1m = cpu_usage / 100.0
                    load_average_5m = cpu_usage / 100.0
                    load_average_15m = cpu_usage / 100.0
                    print(f"      ✅ Load (Windows estimate): {load_average_1m:.2f}")
                
                print("   6. Updating database...")
                # Update system resources in database
                database.update_system_resources(
                    node_id=node_id,
                    cpu_percent=cpu_usage,
                    memory_total=int(memory_total_mb * 1024 * 1024),  # Convert to bytes
                    memory_used=int(memory_used_mb * 1024 * 1024),    # Convert to bytes  
                    disk_total=int(disk_total_gb * 1024 * 1024 * 1024),  # Convert to bytes
                    disk_used=int(disk_used_gb * 1024 * 1024 * 1024),    # Convert to bytes
                    network_sent=network_tx_bytes,
                    network_recv=network_rx_bytes,
                    load_avg=load_average_1m
                )
                print("      ✅ Database updated successfully")
                
                return True
                
            except Exception as e:
                print(f"      ❌ Failed to update system resources: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Test with timeout
        result_container = []
        def worker():
            result = update_system_resources_real()
            result_container.append(result)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        start_time = time.time()
        thread.start()
        
        # Wait with timeout
        thread.join(timeout=20)  # 20 second timeout
        
        if thread.is_alive():
            elapsed = time.time() - start_time
            print(f"❌ SYSTEM RESOURCE COLLECTION HUNG! (timeout after {elapsed:.1f}s)")
            print("   This is likely where the monitoring loop hangs")
            return False
        else:
            if result_container and result_container[0]:
                elapsed = time.time() - start_time
                print(f"✅ System resource collection completed successfully ({elapsed:.2f}s)")
            else:
                print("❌ System resource collection failed")
                return False
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n========================================")
    print("✅ Real system resource test completed!")
    return True

if __name__ == "__main__":
    test_real_system_resources_update()
