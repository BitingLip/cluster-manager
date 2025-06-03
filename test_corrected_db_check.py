#!/usr/bin/env python3
"""
Corrected database activity check with proper column names
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def check_cluster_activity_corrected():
    """Database check with correct column names"""
    
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
    
    print("🔍 Corrected Database Activity Check")
    print(f"⏰ {datetime.now()}")
    print("=" * 60)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        print("\n📊 TABLE RECORD COUNTS:")
        
        tables = ['cluster_nodes', 'system_resources', 'cluster_events', 'gpu_resources']
        for table in tables:
            try:
                result = db.execute_query(f"SELECT COUNT(*) FROM {table}", fetch=True)
                count = result[0][0] if result and result[0] else 0
                print(f"  ✅ {table}: {count} records")
            except Exception as e:
                print(f"  ❌ {table}: {e}")
        
        print("\n🖥️  CLUSTER NODES (with correct column names):")
        try:
            # Use correct column names based on database.py
            node_query = "SELECT id, hostname, node_type, status, last_heartbeat FROM cluster_nodes ORDER BY last_heartbeat DESC"
            nodes = db.execute_query(node_query, fetch=True)
            if nodes:
                print(f"  Found {len(nodes)} nodes:")
                for node in nodes:
                    print(f"  • {node[1]} ({node[0]})")
                    print(f"    Type: {node[2]} | Status: {node[3]} | Heartbeat: {node[4]}")
            else:
                print("  No nodes found")
        except Exception as e:
            print(f"  ❌ Nodes query error: {e}")
        
        print("\n📈 RECENT SYSTEM RESOURCES (Last 10 minutes):")
        ten_min_ago = datetime.now() - timedelta(minutes=10)
        try:
            # Query recent system resources
            resource_query = """
            SELECT node_id, cpu_percent, timestamp 
            FROM system_resources 
            WHERE timestamp >= %s 
            ORDER BY timestamp DESC 
            LIMIT 10
            """
            resources = db.execute_query(resource_query, fetch=True, params=[ten_min_ago])
            if resources:
                print(f"  Found {len(resources)} recent resource entries:")
                for resource in resources:
                    print(f"    • {resource[2]} | Node: {resource[0]} | CPU: {resource[1]}%")
            else:
                print("  No recent resource data found")
        except Exception as e:
            print(f"  ❌ Resources query error: {e}")
        
        print("\n📝 RECENT CLUSTER EVENTS (Last 10 minutes):")
        try:
            # Query recent events
            event_query = """
            SELECT event_type, severity, node_id, timestamp 
            FROM cluster_events 
            WHERE timestamp >= %s 
            ORDER BY timestamp DESC 
            LIMIT 10
            """
            events = db.execute_query(event_query, fetch=True, params=[ten_min_ago])
            if events:
                print(f"  Found {len(events)} recent events:")
                for event in events:
                    print(f"    • {event[3]} | {event[0]} ({event[1]}) | Node: {event[2]}")
            else:
                print("  No recent events found")
        except Exception as e:
            print(f"  ❌ Events query error: {e}")
        
        print("\n🎮 GPU RESOURCES:")
        try:
            gpu_query = "SELECT COUNT(*) FROM gpu_resources"
            gpu_count = db.execute_query(gpu_query, fetch=True)
            count = gpu_count[0][0] if gpu_count else 0
            print(f"  GPU resource entries: {count}")
            
            if count > 0:
                recent_gpu_query = "SELECT node_id, gpu_id, utilization, timestamp FROM gpu_resources ORDER BY timestamp DESC LIMIT 3"
                recent_gpus = db.execute_query(recent_gpu_query, fetch=True)
                if recent_gpus:
                    print("  Recent GPU data:")
                    for gpu in recent_gpus:
                        print(f"    • {gpu[3]} | Node: {gpu[0]} | GPU {gpu[1]} | Util: {gpu[2]}%")
        except Exception as e:
            print(f"  ❌ GPU query error: {e}")
        
        # Real-time monitoring verification
        print("\n🔄 REAL-TIME MONITORING VERIFICATION:")
        
        # Check for very recent activity (last 2 minutes)
        two_min_ago = datetime.now() - timedelta(minutes=2)
        
        try:
            recent_activity_query = """
            SELECT COUNT(*) FROM system_resources WHERE timestamp >= %s
            """
            recent_count = db.execute_query(recent_activity_query, fetch=True, params=[two_min_ago])
            count = recent_count[0][0] if recent_count else 0
            
            if count > 0:
                print(f"  ✅ ACTIVE MONITORING - {count} resource updates in last 2 minutes")
                
                # Get the latest entry
                latest_query = """
                SELECT node_id, cpu_percent, timestamp 
                FROM system_resources 
                ORDER BY timestamp DESC 
                LIMIT 1
                """
                latest = db.execute_query(latest_query, fetch=True)
                if latest:
                    print(f"  📊 Latest: {latest[0][2]} | Node: {latest[0][0]} | CPU: {latest[0][1]}%")
            else:
                print("  ⚠️  NO RECENT MONITORING DATA")
                print("     Cluster manager may not be actively monitoring")
        except Exception as e:
            print(f"  ❌ Recent activity check error: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Database activity check completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_cluster_activity_corrected()
    sys.exit(0 if success else 1)
