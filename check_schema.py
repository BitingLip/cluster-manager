#!/usr/bin/env python3
"""
Check current database schema
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def check_current_schema():
    """Check the current database schema"""
    
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
    
    print("Checking current database schema...")
    
    try:
        # Initialize database connection
        db = ClusterDatabase(config)
        print("✅ Database connection established")
        
        # Check existing tables
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        tables = db.execute_query(tables_query, fetch=True)
        
        print(f"\nFound {len(tables)} tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        # Check cluster_nodes table structure
        if any(t['table_name'] == 'cluster_nodes' for t in tables):
            print("\n📋 cluster_nodes table structure:")
            columns_query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'cluster_nodes' 
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """
            columns = db.execute_query(columns_query, fetch=True)
            
            for col in columns:
                print(f"  - {col['column_name']}: {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema check failed: {e}")
        return False

if __name__ == "__main__":
    success = check_current_schema()
    sys.exit(0 if success else 1)
