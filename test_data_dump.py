#!/usr/bin/env python3
"""
Very simple database data dump to see what's actually in the tables
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def dump_database_data():
    """Simple dump of database data"""
    
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
    
    print("🗂️  Database Data Dump")
    print("=" * 50)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        # Table list to check
        tables = ['cluster_nodes', 'system_resources', 'cluster_events']
        
        for table in tables:
            print(f"\n📋 {table.upper()}:")
            print("-" * 30)
            
            try:
                # Get all data from table
                query = f"SELECT * FROM {table} LIMIT 5"
                result = db.execute_query(query, fetch=True)
                
                if result:
                    print(f"✅ Found {len(result)} rows (showing first 5):")
                    for i, row in enumerate(result):
                        print(f"  Row {i+1}: {row}")
                        print(f"    Type: {type(row)}")
                        if hasattr(row, '__len__'):
                            print(f"    Length: {len(row)}")
                        print()
                else:
                    print("❌ No data found")
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("✅ Data dump completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database dump failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = dump_database_data()
    sys.exit(0 if success else 1)
