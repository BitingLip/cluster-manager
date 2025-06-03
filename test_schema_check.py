#!/usr/bin/env python3
"""
Check actual database schema for cluster manager tables
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def check_database_schema():
    """Check the actual schema of cluster manager tables"""
    
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
    
    print("🔍 Checking actual database schema...")
    print("=" * 60)
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        
        # Get all cluster-related tables
        tables = ['cluster_nodes', 'system_resources', 'cluster_events', 'gpu_resources', 
                 'network_topology', 'resource_allocations', 'load_balancing_metrics', 'cluster_config']
        
        for table in tables:
            print(f"\n📋 TABLE: {table}")
            print("-" * 40)
            
            # Get column information
            schema_query = f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND table_schema = 'public'
            ORDER BY ordinal_position
            """
            
            columns = db.execute_query(schema_query, fetch=True)
              if columns:
                for col in columns:
                    # Handle different return formats
                    if isinstance(col, (list, tuple)) and len(col) >= 3:
                        col_name, data_type, is_nullable = col[0], col[1], col[2]
                        col_default = col[3] if len(col) > 3 else None
                    elif isinstance(col, dict):
                        col_name = col.get('column_name')
                        data_type = col.get('data_type')
                        is_nullable = col.get('is_nullable')
                        col_default = col.get('column_default')
                    else:
                        print(f"  Unexpected column format: {col}")
                        continue
                        
                    nullable = "NULL" if is_nullable == 'YES' else "NOT NULL"
                    default = f" DEFAULT {col_default}" if col_default else ""
                    print(f"  {col_name} ({data_type}) {nullable}{default}")
                
                # Get row count
                count_query = f"SELECT COUNT(*) FROM {table}"
                count_result = db.execute_query(count_query, fetch=True)
                row_count = count_result[0][0] if count_result else 0
                print(f"  📊 Rows: {row_count}")
                
                # Show sample data if rows exist
                if row_count > 0:
                    sample_query = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 3"
                    sample_data = db.execute_query(sample_query, fetch=True)
                    if sample_data:
                        print("  📄 Sample data:")
                        for i, row in enumerate(sample_data):
                            print(f"    Row {i+1}: {row}")
            else:
                print(f"  ❌ Table {table} not found or no columns")
        
        print("\n" + "=" * 60)
        print("✅ Schema check completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_database_schema()
    sys.exit(0 if success else 1)
