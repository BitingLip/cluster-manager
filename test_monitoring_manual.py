#!/usr/bin/env python3
"""
Test the cluster manager monitoring loop manually
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add app directory to path
app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(app_path))

# Load environment variables
env_file = Path(__file__).parent / "config" / "cluster-manager-db.env"
if env_file.exists():
    load_dotenv(env_file)

from simple_config import ClusterManagerSettings
from database import ClusterDatabase

def test_monitoring_manually():
    """Manually test the monitoring functions that should run in the loop"""
    print("🔄 Testing Cluster Manager Monitoring Loop")
    print("=" * 60)
    
    try:
        # Initialize like the cluster manager does
        settings = ClusterManagerSettings()
        db_config = {
            'CLUSTER_DB_HOST': settings.get_config_value('CLUSTER_DB_HOST', 'localhost'),
            'CLUSTER_DB_PORT': settings.get_config_value('CLUSTER_DB_PORT', '5432'),
            'CLUSTER_DB_NAME': settings.get_config_value('CLUSTER_DB_NAME', 'bitinglip_cluster'),
            'CLUSTER_DB_USER': settings.get_config_value('CLUSTER_DB_USER', 'postgres'),
            'CLUSTER_DB_PASSWORD': settings.get_config_value('CLUSTER_DB_PASSWORD', 'password'),
        }
        
        database = ClusterDatabase(db_config)
        print("✅ Database connection established")
        
        # Get a test node ID (use one of the running managers)
        nodes = database.list_nodes()
        if not nodes:
            print("❌ No nodes found in database")
            return False
            
        test_node_id = nodes[0]['id']
        print(f"🎯 Using test node: {test_node_id}")
        
        # Test system resource update (simulate what _update_system_resources does)
        print("\n📊 Testing System Resource Update")
        import psutil
        
        # Gather system data like the cluster manager does
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        memory = psutil.virtual_memory()
        memory_total_mb = memory.total / (1024 * 1024)
        memory_used_mb = memory.used / (1024 * 1024)
        
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        
        net_io = psutil.net_io_counters()
        network_rx_bytes = net_io.bytes_recv
        network_tx_bytes = net_io.bytes_sent
        
        # Windows-compatible load average
        try:
            load_avg = os.getloadavg()
            load_average_1m = load_avg[0]
        except (AttributeError, OSError):
            load_average_1m = cpu_usage / 100.0
        
        print(f"  CPU: {cpu_usage:.1f}%")
        print(f"  Memory: {memory_used_mb:.0f}MB / {memory_total_mb:.0f}MB")
        print(f"  Load Average: {load_average_1m:.2f}")
        
        # Test the database update (this is what might be failing)
        print("\n💾 Testing Database Update")
        success = database.update_system_resources(
            node_id=test_node_id,
            cpu_percent=cpu_usage,
            memory_total=int(memory_total_mb * 1024 * 1024),  # Convert to bytes
            memory_used=int(memory_used_mb * 1024 * 1024),    # Convert to bytes  
            disk_total=int(disk_total_gb * 1024 * 1024 * 1024),  # Convert to bytes
            disk_used=int(disk_used_gb * 1024 * 1024 * 1024),    # Convert to bytes
            network_sent=network_tx_bytes,
            network_recv=network_rx_bytes,
            load_avg=load_average_1m
        )
        
        if success:
            print("✅ System resources updated successfully")
            
            # Verify the data was written
            check_query = "SELECT COUNT(*) as count FROM system_resources WHERE node_id = %s"
            result = database.execute_query(check_query, (test_node_id,), fetch=True)
            count = result[0]['count'] if result else 0
            print(f"✅ System resource records for this node: {count}")
            
        else:
            print("❌ Failed to update system resources")
        
        # Test GPU resource update (might also be needed)
        print("\n🎮 Testing GPU Resource Update")
        # Simple test with empty GPU data (most systems won't have NVIDIA GPUs)
        gpu_success = database.update_gpu_resources(test_node_id, [])
        if gpu_success:
            print("✅ GPU resources updated successfully (empty list)")
        else:
            print("❌ Failed to update GPU resources")
        
        print("\n🎯 Monitoring Test Results:")
        if success:
            print("✅ System resource monitoring: WORKING")
            print("✅ Database updates: WORKING") 
            print("✅ Monitoring loop should work correctly")
        else:
            print("❌ System resource monitoring: FAILED")
            print("❌ Issue found in database update")
        
        return success
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'database' in locals():
            database.close()

if __name__ == "__main__":
    test_monitoring_manually()
