#!/usr/bin/env python3
"""
Quick check for new monitoring data after the fix
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def quick_monitoring_check():
    """Quick check for recent monitoring activity"""
    
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
    
    print("🔍 Quick Monitoring Check")
    print(f"⏰ {datetime.now()}")
    print("=" * 40)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        # Check for new nodes (last 2 minutes)
        two_min_ago = datetime.now() - timedelta(minutes=2)
        
        print("\n📊 RECENT ACTIVITY:")
        
        # New nodes
        node_query = "SELECT id, hostname, joined_at FROM cluster_nodes WHERE joined_at >= %s ORDER BY joined_at DESC"
        recent_nodes = db.execute_query(node_query, fetch=True, params=[two_min_ago])
        print(f"  New nodes: {len(recent_nodes) if recent_nodes else 0}")
        if recent_nodes:
            for node in recent_nodes:
                print(f"    • {node['hostname']} ({node['id']}) - {node['joined_at']}")
        
        # New system resources
        resource_query = "SELECT node_id, cpu_percent, timestamp FROM system_resources WHERE timestamp >= %s ORDER BY timestamp DESC"
        recent_resources = db.execute_query(resource_query, fetch=True, params=[two_min_ago])
        print(f"  New resources: {len(recent_resources) if recent_resources else 0}")
        if recent_resources:
            for resource in recent_resources:
                print(f"    • {resource['node_id']} - CPU: {resource['cpu_percent']}% at {resource['timestamp']}")
        
        # New events
        event_query = "SELECT event_type, node_id, timestamp FROM cluster_events WHERE timestamp >= %s ORDER BY timestamp DESC"
        recent_events = db.execute_query(event_query, fetch=True, params=[two_min_ago])
        print(f"  New events: {len(recent_events) if recent_events else 0}")
        if recent_events:
            for event in recent_events:
                print(f"    • {event['event_type']} from {event['node_id']} at {event['timestamp']}")
        
        # Overall monitoring status
        print(f"\n🎯 STATUS:")
        if recent_resources:
            print("  ✅ MONITORING IS ACTIVE - New resource data detected!")
        elif recent_nodes:
            print("  🔄 CLUSTER MANAGER RESTARTED - Waiting for monitoring data...")
        else:
            print("  ⚠️  NO NEW MONITORING DATA - Check if monitoring loop is working")
        
        print("\n" + "=" * 40)
        print("✅ Quick check completed!")
        
        return len(recent_resources) > 0 if recent_resources else False
        
    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_monitoring_check()
    sys.exit(0 if success else 1)
