#!/usr/bin/env python3
"""
Simple cluster manager test without complex imports
"""

import sys
import os
import time
import socket
import threading
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

# Load environment variables
env_file = Path(__file__).parent / "config" / "cluster-manager-db.env"
if env_file.exists():
    import dotenv
    dotenv.load_dotenv(env_file)

from database import ClusterDatabase

def simple_cluster_test():
    """Simple test of cluster manager functionality"""
    print("🚀 BitingLip Cluster Manager - Simple Test")
    print("=" * 50)
    
    # Database configuration
    db_config = {
        'CLUSTER_DB_HOST': os.getenv('CLUSTER_DB_HOST', 'localhost'),
        'CLUSTER_DB_PORT': os.getenv('CLUSTER_DB_PORT', '5432'),
        'CLUSTER_DB_NAME': os.getenv('CLUSTER_DB_NAME', 'bitinglip_cluster'),
        'CLUSTER_DB_USER': os.getenv('CLUSTER_DB_USER', 'cluster_manager'),
        'CLUSTER_DB_PASSWORD': os.getenv('CLUSTER_DB_PASSWORD', 'cluster_manager_2025!'),
        'CLUSTER_DB_MIN_CONN': os.getenv('CLUSTER_DB_MIN_CONN', '2'),
        'CLUSTER_DB_MAX_CONN': os.getenv('CLUSTER_DB_MAX_CONN', '10')
    }
    
    try:
        # Initialize database
        print("📊 Initializing database connection...")
        db = ClusterDatabase(db_config)
        print("✅ Database connected successfully")
        
        # Register this node as cluster manager
        node_id = "cluster-manager-" + socket.gethostname()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        port = 8002
        
        print(f"🏷️  Registering node: {node_id}")
        print(f"   Hostname: {hostname}")
        print(f"   IP: {ip_address}")
        print(f"   Port: {port}")
        
        success = db.register_node(
            node_id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=port,
            node_type='manager',
            capabilities=['cluster_management', 'resource_monitoring', 'task_scheduling'],
            metadata={'version': '1.0.0', 'role': 'cluster_manager'}
        )
        
        if success:
            print("✅ Node registered successfully")
        else:
            print("❌ Node registration failed")
            return
        
        # Test cluster health monitoring
        print("\n🏥 Checking cluster health...")
        health = db.get_cluster_health()
        print(f"Total nodes: {health.get('total_nodes', 0)}")
        print(f"Node status: {health.get('node_status', {})}")
        
        # Test event logging
        print("\n📝 Logging cluster events...")
        db.log_cluster_event(
            event_type='cluster_manager_startup',
            event_data={'node_id': node_id, 'startup_time': time.time()},
            severity='info'
        )
        print("✅ Event logged successfully")
        
        # List all nodes
        print("\n📋 Current cluster nodes:")
        nodes = db.list_nodes()
        for node in nodes:
            print(f"  - {node['id']} ({node['hostname']}) - {node['status']}")
        
        print("\n🎉 Cluster Manager test completed successfully!")
        print("🔄 Starting heartbeat monitoring (press Ctrl+C to stop)...")
        
        # Start heartbeat loop
        try:
            while True:
                # Update heartbeat
                db.update_node_heartbeat(node_id)
                
                # Get and display cluster status
                nodes = db.list_nodes(status='online')
                online_count = len(nodes)
                
                print(f"💓 Heartbeat sent - {online_count} nodes online")
                
                time.sleep(30)  # Heartbeat every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down cluster manager...")
            
            # Mark node as offline
            db.mark_node_offline(node_id)
            
            # Log shutdown event
            db.log_cluster_event(
                event_type='cluster_manager_shutdown',
                event_data={'node_id': node_id, 'shutdown_time': time.time()},
                severity='info'
            )
            
            print("✅ Cluster manager stopped cleanly")
        
    except Exception as e:
        print(f"❌ Cluster manager test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_cluster_test()
