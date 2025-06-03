#!/usr/bin/env python3
"""
Test database connection and basic operations
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

from simple_config import ClusterManagerSettings
from database import ClusterDatabase

def test_database_connection():
    """Test database connection and basic operations"""
    print("🔧 Testing Database Connection...")
    print("=" * 50)
    
    try:
        # Load configuration
        settings = ClusterManagerSettings()
        
        # Initialize database
        database = ClusterDatabase(
            host=settings.get_config_value('CLUSTER_DB_HOST', 'localhost'),
            port=int(settings.get_config_value('CLUSTER_DB_PORT', '5432')),
            database=settings.get_config_value('CLUSTER_DB_NAME', 'bitinglip_cluster'),
            user=settings.get_config_value('CLUSTER_DB_USER', 'postgres'),
            password=settings.get_config_value('CLUSTER_DB_PASSWORD', 'password'),
            min_connections=1,
            max_connections=5
        )
        print("✅ Database connection pool created successfully")
        
        # Initialize schema
        if database.initialize_schema():
            print("✅ Database schema initialized successfully")
        else:
            print("❌ Failed to initialize database schema")
            return False
        
        # Test basic operations
        test_node_id = "test_node_001"
        
        # Test node registration
        if database.register_node(
            node_id=test_node_id,
            node_type="test",
            host="localhost",
            port=8001,
            metadata={"test": True}
        ):
            print("✅ Node registration successful")
        else:
            print("❌ Node registration failed")
            return False
        
        # Test heartbeat update
        if database.update_node_heartbeat(test_node_id):
            print("✅ Heartbeat update successful")
        else:
            print("❌ Heartbeat update failed")
            return False
        
        # Test system resources update
        if database.update_system_resources(
            node_id=test_node_id,
            cpu_usage=25.5,
            memory_usage=60.2,
            disk_usage=45.8,
            gpu_usage=None
        ):
            print("✅ System resources update successful")
        else:
            print("❌ System resources update failed")
            return False
        
        # Test queries
        active_nodes = database.get_active_nodes()
        print(f"✅ Found {len(active_nodes)} active nodes")
        
        node_resources = database.get_node_resources(test_node_id)
        if node_resources:
            print("✅ Node resources retrieved successfully")
            print(f"   CPU: {node_resources['cpu_usage']}%")
            print(f"   Memory: {node_resources['memory_usage']}%")
            print(f"   Disk: {node_resources['disk_usage']}%")
        else:
            print("❌ Failed to retrieve node resources")
            return False
        
        # Cleanup
        database.close()
        print("✅ Database connection closed")
        
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_database_connection()
