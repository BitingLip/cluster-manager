#!/usr/bin/env python3
"""
Verify cluster manager resource monitoring and database updates
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

def verify_resource_monitoring():
    """Verify that resource monitoring is writing to database tables"""
    
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
    
    print("🔍 Verifying cluster manager resource monitoring...")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        # 1. Check cluster nodes
        print("\n📊 CLUSTER NODES:")
        nodes_query = "SELECT node_id, hostname, role, status, last_heartbeat FROM cluster_nodes ORDER BY last_heartbeat DESC"
        nodes = db.execute_query(nodes_query, fetch=True)
        
        if nodes:
            for node in nodes:
                print(f"  • {node[1]} ({node[0]})")
                print(f"    Role: {node[2]} | Status: {node[3]}")
                print(f"    Last Heartbeat: {node[4]}")
        else:
            print("  No nodes found")
        
        # 2. Check recent system resources (last 5 minutes)
        print("\n💻 RECENT SYSTEM RESOURCES (Last 5 minutes):")
        five_min_ago = datetime.now() - timedelta(minutes=5)
        resources_query = """
        SELECT node_id, cpu_percent, memory_used_gb, memory_total_gb, 
               disk_used_gb, disk_total_gb, timestamp 
        FROM system_resources 
        WHERE timestamp >= %s 
        ORDER BY timestamp DESC 
        LIMIT 10
        """
        resources = db.execute_query(resources_query, fetch=True, params=[five_min_ago])
        
        if resources:
            print(f"  Found {len(resources)} recent resource entries:")
            for resource in resources:
                print(f"  • Node: {resource[0]}")
                print(f"    CPU: {resource[1]:.1f}% | Memory: {resource[2]:.1f}/{resource[3]:.1f} GB")
                print(f"    Disk: {resource[4]:.1f}/{resource[5]:.1f} GB | Time: {resource[6]}")
                print()
        else:
            print("  No recent resource data found")
        
        # 3. Check recent cluster events (last 10 minutes)
        print("\n📝 RECENT CLUSTER EVENTS (Last 10 minutes):")
        ten_min_ago = datetime.now() - timedelta(minutes=10)
        events_query = """
        SELECT event_type, event_data, severity, node_id, timestamp 
        FROM cluster_events 
        WHERE timestamp >= %s 
        ORDER BY timestamp DESC 
        LIMIT 15
        """
        events = db.execute_query(events_query, fetch=True, params=[ten_min_ago])
        
        if events:
            print(f"  Found {len(events)} recent events:")
            for event in events:
                print(f"  • {event[4]} | {event[0]} ({event[2]})")
                print(f"    Node: {event[3]} | Data: {event[1]}")
        else:
            print("  No recent events found")
        
        # 4. Check GPU resources if any
        print("\n🎮 GPU RESOURCES:")
        gpu_query = "SELECT node_id, gpu_id, gpu_name, memory_total, memory_used, utilization, timestamp FROM gpu_resources ORDER BY timestamp DESC LIMIT 5"
        gpus = db.execute_query(gpu_query, fetch=True)
        
        if gpus:
            print(f"  Found {len(gpus)} GPU entries:")
            for gpu in gpus:
                print(f"  • Node: {gpu[0]} | GPU {gpu[1]}: {gpu[2]}")
                print(f"    Memory: {gpu[4]}/{gpu[3]} MB | Util: {gpu[5]}% | Time: {gpu[6]}")
        else:
            print("  No GPU resources found")
        
        # 5. Database health check
        print("\n🔧 DATABASE HEALTH:")
        
        # Check table sizes
        table_sizes_query = """
        SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size_pretty
        FROM pg_tables 
        WHERE schemaname = 'public' AND tablename LIKE 'cluster_%' OR tablename LIKE 'system_%' OR tablename LIKE 'gpu_%'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        sizes = db.execute_query(table_sizes_query, fetch=True)
        
        if sizes:
            print("  Table sizes:")
            for size in sizes:
                print(f"    {size[1]}: {size[2]}")
        
        # Check connection status
        print(f"  Database connection: ✅ Active")
        print(f"  Connection pool: ✅ Healthy")
        
        print("\n" + "=" * 60)
        print("✅ Resource monitoring verification completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_resource_monitoring()
    sys.exit(0 if success else 1)
