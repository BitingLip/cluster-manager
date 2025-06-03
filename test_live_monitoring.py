#!/usr/bin/env python3
"""
Real-time monitoring verification - check if cluster manager is writing data
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def verify_live_monitoring():
    """Verify that the cluster manager is actively writing monitoring data"""
    
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
    
    print("🔍 Real-time Cluster Manager Monitoring Verification")
    print(f"⏰ Started at: {datetime.now()}")
    print("=" * 70)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        # Capture initial state
        print("\n📊 INITIAL STATE:")
        
        # Count current records
        initial_counts = {}
        tables = ['cluster_nodes', 'system_resources', 'cluster_events']
        
        for table in tables:
            try:
                count_result = db.execute_query(f"SELECT COUNT(*) FROM {table}", fetch=True)
                initial_counts[table] = count_result[0][0] if count_result else 0
                print(f"  {table}: {initial_counts[table]} records")
            except Exception as e:
                print(f"  {table}: Error - {e}")
                initial_counts[table] = 0
        
        # Get latest timestamps
        print("\n🕐 LATEST TIMESTAMPS:")
        for table in ['system_resources', 'cluster_events']:
            try:
                latest_query = f"SELECT MAX(timestamp) FROM {table}"
                latest_result = db.execute_query(latest_query, fetch=True)
                latest_time = latest_result[0][0] if latest_result and latest_result[0][0] else "No data"
                print(f"  {table}: {latest_time}")
            except Exception as e:
                print(f"  {table}: Error - {e}")
        
        # Wait and monitor for changes
        print(f"\n⏳ Waiting 60 seconds to observe new data...")
        print("   (The cluster manager should write monitoring data every ~30 seconds)")
        
        for i in range(6):  # 6 x 10 seconds = 60 seconds
            time.sleep(10)
            dots = "." * (i + 1)
            print(f"   Monitoring{dots} ({(i+1)*10}s)", end="\r")
        
        print(f"\n\n📈 STATE AFTER 60 SECONDS:")
        
        # Check for new records
        new_data_detected = False
        
        for table in tables:
            try:
                count_result = db.execute_query(f"SELECT COUNT(*) FROM {table}", fetch=True)
                current_count = count_result[0][0] if count_result else 0
                
                change = current_count - initial_counts[table]
                if change > 0:
                    print(f"  ✅ {table}: {current_count} records (+{change} new)")
                    new_data_detected = True
                else:
                    print(f"  ⚪ {table}: {current_count} records (no change)")
                    
            except Exception as e:
                print(f"  ❌ {table}: Error - {e}")
        
        # Check recent activity (last 2 minutes)
        print(f"\n🔍 RECENT ACTIVITY (Last 2 minutes):")
        two_min_ago = datetime.now() - timedelta(minutes=2)
        
        # Recent system resources
        try:
            recent_resources_query = "SELECT COUNT(*) FROM system_resources WHERE timestamp >= %s"
            recent_resources = db.execute_query(recent_resources_query, fetch=True, params=[two_min_ago])
            resource_count = recent_resources[0][0] if recent_resources else 0
            print(f"  System resources: {resource_count} new entries")
            
            if resource_count > 0:
                # Show latest resource entry
                latest_resource_query = "SELECT * FROM system_resources ORDER BY timestamp DESC LIMIT 1"
                latest_resource = db.execute_query(latest_resource_query, fetch=True)
                if latest_resource:
                    print(f"    Latest entry: {latest_resource[0]}")
                    
        except Exception as e:
            print(f"  System resources: Error - {e}")
        
        # Recent events
        try:
            recent_events_query = "SELECT COUNT(*) FROM cluster_events WHERE timestamp >= %s"
            recent_events = db.execute_query(recent_events_query, fetch=True, params=[two_min_ago])
            event_count = recent_events[0][0] if recent_events else 0
            print(f"  Cluster events: {event_count} new entries")
            
            if event_count > 0:
                # Show latest event
                latest_event_query = "SELECT * FROM cluster_events ORDER BY timestamp DESC LIMIT 1"
                latest_event = db.execute_query(latest_event_query, fetch=True)
                if latest_event:
                    print(f"    Latest event: {latest_event[0]}")
                    
        except Exception as e:
            print(f"  Cluster events: Error - {e}")
        
        # Overall assessment
        print(f"\n🎯 MONITORING ASSESSMENT:")
        if new_data_detected:
            print("  ✅ LIVE MONITORING ACTIVE - New data is being written!")
            print("  ✅ Cluster manager is successfully persisting monitoring data")
            print("  ✅ Database integration is working correctly")
        else:
            print("  ⚠️  NO NEW DATA DETECTED")
            print("  🔍 Cluster manager may not be running or monitoring loop may be inactive")
            print("  💡 Check if the cluster manager service is running")
        
        print("\n" + "=" * 70)
        print("✅ Live monitoring verification completed!")
        
        return new_data_detected
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_live_monitoring()
    sys.exit(0 if success else 1)
