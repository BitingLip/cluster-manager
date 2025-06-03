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

def start_cluster_manager():
    """Start the cluster manager service"""
    print("🚀 Starting BitingLip Cluster Manager...")
    print("=" * 50)
    
    try:
        # Initialize cluster manager
        manager = ClusterManager()
        print("✅ Cluster Manager initialized successfully")
        
        # Start the service
        print("🔄 Starting cluster monitoring loop...")
        manager.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Cluster Manager stopped by user")
    except Exception as e:
        print(f"❌ Cluster Manager failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_cluster_manager()
