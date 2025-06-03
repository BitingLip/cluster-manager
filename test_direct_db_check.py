#!/usr/bin/env python3
"""
Direct database query to check cluster manager activity
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def check_cluster_activity():
    """Direct database check for cluster manager activity"""
    
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
    
    print("🔍 Direct Database Activity Check")
    print(f"⏰ {datetime.now()}")
    print("=" * 50)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        # Simple raw SQL queries to check data
        print("\n📊 RAW TABLE COUNTS:")
        
        tables = ['cluster_nodes', 'system_resources', 'cluster_events', 'gpu_resources']
        for table in tables:
            try:
                result = db.execute_query(f"SELECT COUNT(*) FROM {table}", fetch=True)
                count = result[0][0] if result and result[0] else 0
                print(f"  {table}: {count} records")
            except Exception as e:
                print(f"  {table}: ERROR - {e}")
        
        print("\n📅 RECENT DATA (Last 5 minutes):")
        five_min_ago = datetime.now() - timedelta(minutes=5)
        
        # Check recent events
        try:
            event_query = """
            SELECT event_type, event_data, severity, node_id, timestamp 
            FROM cluster_events 
            WHERE timestamp >= %s 
            ORDER BY timestamp DESC
            """
            events = db.execute_query(event_query, fetch=True, params=[five_min_ago])
            print(f"  Events: {len(events) if events else 0} recent entries")
            if events:
                for event in events[:3]:  # Show first 3
                    print(f"    • {event[4]} | {event[0]} ({event[2]})")
        except Exception as e:
            print(f"  Events: ERROR - {e}")
        
        # Check recent system resources
        try:
            resource_query = """
            SELECT node_id, cpu_percent, timestamp 
            FROM system_resources 
            WHERE timestamp >= %s 
            ORDER BY timestamp DESC
            """
            resources = db.execute_query(resource_query, fetch=True, params=[five_min_ago])
            print(f"  Resources: {len(resources) if resources else 0} recent entries")
            if resources:
                for resource in resources[:3]:  # Show first 3
                    print(f"    • {resource[2]} | Node: {resource[0]} | CPU: {resource[1]}%")
        except Exception as e:
            print(f"  Resources: ERROR - {e}")
        
        # Check cluster nodes status
        print("\n🖥️  CLUSTER NODES:")
        try:
            node_query = "SELECT id, hostname, role, status, last_heartbeat FROM cluster_nodes ORDER BY last_heartbeat DESC"
            nodes = db.execute_query(node_query, fetch=True)
            if nodes:
                for node in nodes:
                    print(f"  • {node[1]} ({node[0]})")
                    print(f"    Role: {node[2]} | Status: {node[3]} | Heartbeat: {node[4]}")
            else:
                print("  No nodes found")
        except Exception as e:
            print(f"  Nodes: ERROR - {e}")
        
        print("\n✅ Direct database check completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_cluster_activity()
    sys.exit(0 if success else 1)
