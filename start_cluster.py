#!/usr/bin/env python3
"""
Start the cluster manager service
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

# Load environment variables
env_file = Path(__file__).parent / "config" / "cluster-manager-db.env"
if env_file.exists():
    import dotenv
    dotenv.load_dotenv(env_file)

from cluster_manager import ClusterManager
import signal
import time

def start_cluster_manager():
    """Start the cluster manager service"""
    print("🚀 Starting BitingLip Cluster Manager...")
    print("=" * 50)
    
    manager = None
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Received shutdown signal, stopping cluster manager...")
        if manager:
            manager.stop()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize cluster manager
        manager = ClusterManager()
        print("✅ Cluster Manager initialized successfully")
        
        # Start the service
        print("🔄 Starting cluster monitoring loop...")
        if manager.start():
            print("✅ Cluster Manager started successfully")
            print("🔄 Monitoring active - Press Ctrl+C to stop")
            
            # Keep main thread alive while monitoring runs
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Cluster Manager stopped by user")
                manager.stop()
        else:
            print("❌ Failed to start cluster manager")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Cluster Manager failed: {e}")
        import traceback
        traceback.print_exc()
        if manager:
            manager.stop()
        sys.exit(1)

if __name__ == "__main__":
    start_cluster_manager()
