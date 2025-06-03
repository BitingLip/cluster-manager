#!/usr/bin/env python3
"""
Simple database schema and data check
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def simple_schema_check():
    """Simple check of database schema and data"""
    
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
    
    print("🔍 Simple database check...")
    print("=" * 60)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        # Simple table checks
        tables = ['cluster_nodes', 'system_resources', 'cluster_events']
        
        for table in tables:
            print(f"\n📋 {table.upper()}:")
            
            # Check columns
            try:
                result = db.execute_query(f"SELECT * FROM {table} LIMIT 1", fetch=True)
                print("✅ Table exists and accessible")
                
                # Get row count
                count_result = db.execute_query(f"SELECT COUNT(*) FROM {table}", fetch=True)
                row_count = count_result[0][0] if count_result else 0
                print(f"📊 Total rows: {row_count}")
                
                # Show recent data if any
                if row_count > 0:
                    recent_data = db.execute_query(f"SELECT * FROM {table} LIMIT 2", fetch=True)
                    print("📄 Sample rows:")
                    for i, row in enumerate(recent_data):
                        print(f"  Row {i+1}: {row}")
                        
            except Exception as e:
                print(f"❌ Error accessing {table}: {e}")
        
        # Test specific queries that cluster manager uses
        print(f"\n🔧 TESTING CLUSTER MANAGER QUERIES:")
        
        # Test node registration
        try:
            node_query = "SELECT * FROM cluster_nodes LIMIT 3"
            nodes = db.execute_query(node_query, fetch=True)
            print(f"✅ Node query works - found {len(nodes) if nodes else 0} nodes")
        except Exception as e:
            print(f"❌ Node query failed: {e}")
        
        # Test system resources
        try:
            resource_query = "SELECT * FROM system_resources ORDER BY timestamp DESC LIMIT 3"
            resources = db.execute_query(resource_query, fetch=True)
            print(f"✅ Resource query works - found {len(resources) if resources else 0} entries")
        except Exception as e:
            print(f"❌ Resource query failed: {e}")
        
        # Test cluster events
        try:
            event_query = "SELECT * FROM cluster_events ORDER BY timestamp DESC LIMIT 3"
            events = db.execute_query(event_query, fetch=True)
            print(f"✅ Event query works - found {len(events) if events else 0} events")
        except Exception as e:
            print(f"❌ Event query failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Simple database check completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_schema_check()
    sys.exit(0 if success else 1)
