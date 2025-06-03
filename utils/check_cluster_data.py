#!/usr/bin/env python3
"""
Check cluster manager data in the database
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(cluster_app_path))

# Load environment variables
env_file = Path(__file__).parent.parent / "config" / "cluster-manager-db.env"
if env_file.exists():
    import dotenv
    dotenv.load_dotenv(env_file)

from simple_config import ClusterManagerSettings
from database import ClusterDatabase

def check_cluster_data():
    """Check current cluster data in database"""
    print("📊 Checking Cluster Data in Database...")
    print("=" * 50)
    
    try:
        # Load configuration
        settings = ClusterManagerSettings()
        
        # Initialize database
        database = ClusterDatabase(
            host=settings.db_host,
            port=int(settings.db_port),
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password
        )
        
        # Get active nodes
        print("🔍 Active Nodes:")
        active_nodes = database.get_active_nodes()
        for node in active_nodes:
            print(f"   📍 {node['node_id']}")
            print(f"      Type: {node['node_type']}")
            print(f"      Host: {node['host']}:{node['port']}")
            print(f"      Status: {node['status']}")
            print(f"      Last Heartbeat: {node['last_heartbeat']}")
            
            # Get resources for this node
            resources = database.get_node_resources(node['node_id'])
            if resources:
                print(f"      CPU: {resources['cpu_usage']:.1f}%")
                print(f"      Memory: {resources['memory_usage']:.1f}%")
                print(f"      Disk: {resources['disk_usage']:.1f}%")
                print(f"      Last Updated: {resources['timestamp']}")
            print()
        
        # Get cluster health
        print("🏥 Cluster Health:")
        health = database.get_cluster_health()
        print(f"   Node Counts: {health.get('node_counts', {})}")
        if 'average_resources' in health:
            avg = health['average_resources']
            if avg.get('avg_cpu') is not None:
                print(f"   Average CPU: {avg['avg_cpu']:.1f}%")
                print(f"   Average Memory: {avg['avg_memory']:.1f}%")
                print(f"   Average Disk: {avg['avg_disk']:.1f}%")
        
        database.close()
        print("✅ Database check completed")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_cluster_data()
