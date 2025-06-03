#!/usr/bin/env python3
"""
Test database connection for cluster manager
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def test_database_connection():
    """Test PostgreSQL connection and basic operations"""
    
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
    
    print("Testing cluster database connection...")
    print(f"Host: {config['CLUSTER_DB_HOST']}")
    print(f"Database: {config['CLUSTER_DB_NAME']}")
    print(f"User: {config['CLUSTER_DB_USER']}")
    
    try:
        # Initialize database
        db = ClusterDatabase(config)
        print("✅ Database connection pool initialized successfully")
        
        # Test basic query
        test_query = "SELECT version() as postgres_version"
        result = db.execute_query(test_query, fetch=True)
        
        if result:
            print("✅ Database query successful")
            print(f"PostgreSQL Version: {result[0]['postgres_version']}")
          # Check if cluster tables exist
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'cluster_%'"
        
        tables = db.execute_query(tables_query, fetch=True)
        
        if tables and len(tables) > 0:
            print(f"✅ Found {len(tables)} cluster tables:")
            for table in tables:
                print(f"  - {table['table_name']}")
        else:
            print("⚠️  No cluster tables found. Database schema may need to be initialized.")
            
        # Check all tables in the database
        all_tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        all_tables = db.execute_query(all_tables_query, fetch=True)
        
        if all_tables and len(all_tables) > 0:
            print(f"Found {len(all_tables)} total tables:")
            for table in all_tables:
                print(f"  - {table['table_name']}")
        else:
            print("No tables found in the database.")
            
        print("\n🎉 Database connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_database_connection()
    sys.exit(0 if success else 1)
