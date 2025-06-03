#!/usr/bin/env python3
"""
Test individual monitoring components to identify the issue
"""

import sys
import os
import time
import psutil
from pathlib import Path
from datetime import datetime

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def test_monitoring_components():
    """Test each monitoring component individually"""
    
    # Load environment variables from cluster manager config
    env_file = Path(__file__).parent / "config" / "cluster-manager-db.env"
    
    # Load environment variables
    if env_file.exists():
        import dotenv
        dotenv.load_dotenv(env_file)
    
    config = {
        'CLUSTER_DB_HOST': os.getenv('CLUSTER_DB_HOST', 'localhost'),
        'CLUSTER_DB_PORT': os.getenv('CLUSTER_DB_PORT', '5432'),
        'CLUSTER_DB_NAME': os.getenv('CLUSTER_DB_NAME', 'bitinglip_cluster'),
        'CLUSTER_DB_USER': os.getenv('CLUSTER_DB_USER', 'cluster_manager'),
        'CLUSTER_DB_PASSWORD': os.getenv('CLUSTER_DB_PASSWORD', 'cluster_manager_2025!'),
        'CLUSTER_DB_MIN_CONN': os.getenv('CLUSTER_DB_MIN_CONN', '2'),
        'CLUSTER_DB_MAX_CONN': os.getenv('CLUSTER_DB_MAX_CONN', '10')
    }
    
    print("🔧 Testing Monitoring Components")
    print(f"⏰ {datetime.now()}")
    print("=" * 50)
    
    try:
        # Initialize database
        print("\n1️⃣ Testing database connection...")
        db = ClusterDatabase(config)
        print("✅ Database connection successful")
        
        # Test heartbeat update
        print("\n2️⃣ Testing heartbeat update...")
        test_node_id = "test_node_monitoring_check"
        try:
            result = db.update_node_heartbeat(test_node_id)
            print(f"✅ Heartbeat update result: {result}")
        except Exception as e:
            print(f"❌ Heartbeat update failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test system resource collection
        print("\n3️⃣ Testing system resource collection...")
        try:
            # Get CPU information
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            print(f"✅ CPU: {cpu_count} cores, {cpu_usage}% usage")
            
            # Get memory information
            memory = psutil.virtual_memory()
            memory_total_mb = memory.total // (1024 * 1024)
            memory_used_mb = memory.used // (1024 * 1024)
            print(f"✅ Memory: {memory_used_mb}/{memory_total_mb} MB")
            
            # Test disk information with Windows fix
            if sys.platform == 'win32':
                disk = psutil.disk_usage('C:')
            else:
                disk = psutil.disk_usage('/')
            disk_total_gb = disk.total // (1024 * 1024 * 1024)
            disk_used_gb = disk.used // (1024 * 1024 * 1024)
            print(f"✅ Disk: {disk_used_gb}/{disk_total_gb} GB")
            
            # Get network information
            net_io = psutil.net_io_counters()
            print(f"✅ Network: RX {net_io.bytes_recv}, TX {net_io.bytes_sent}")
            
        except Exception as e:
            print(f"❌ System resource collection failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test system resource database update
        print("\n4️⃣ Testing system resource database update...")
        try:
            result = db.update_system_resources(
                node_id=test_node_id,
                cpu_percent=25.5,
                memory_total=int(8 * 1024 * 1024 * 1024),  # 8GB in bytes
                memory_used=int(4 * 1024 * 1024 * 1024),   # 4GB in bytes
                disk_total=int(500 * 1024 * 1024 * 1024),  # 500GB in bytes
                disk_used=int(200 * 1024 * 1024 * 1024),   # 200GB in bytes
                network_sent=1000000,
                network_recv=2000000,
                load_avg=0.5
            )
            print(f"✅ System resource update result: {result}")
        except Exception as e:
            print(f"❌ System resource update failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test event logging
        print("\n5️⃣ Testing event logging...")
        try:
            result = db.log_cluster_event(
                event_type='monitoring_test',
                event_data={'message': 'Testing monitoring components'},
                severity='info',
                node_id=test_node_id
            )
            print(f"✅ Event logging result: {result}")
        except Exception as e:
            print(f"❌ Event logging failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Check if data was written
        print("\n6️⃣ Verifying data was written...")
        try:
            # Check for our test entries
            resource_check = db.execute_query("SELECT COUNT(*) FROM system_resources WHERE node_id = %s", fetch=True, params=[test_node_id])
            event_check = db.execute_query("SELECT COUNT(*) FROM cluster_events WHERE node_id = %s", fetch=True, params=[test_node_id])
            
            resource_count = resource_check[0]['count'] if resource_check else 0
            event_count = event_check[0]['count'] if event_check else 0
            
            print(f"✅ Test resources written: {resource_count}")
            print(f"✅ Test events written: {event_count}")
        except Exception as e:
            print(f"❌ Data verification failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("✅ Component testing completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_monitoring_components()
    sys.exit(0 if success else 1)
