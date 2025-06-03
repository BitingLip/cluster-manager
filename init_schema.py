#!/usr/bin/env python3
"""
Initialize cluster manager database schema
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def initialize_schema():
    """Initialize the database schema"""
    
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
    
    print("Initializing cluster manager database schema...")
    
    try:
        # Initialize database connection
        db = ClusterDatabase(config)
        print("✅ Database connection established")
        
        # Read schema file
        schema_file = Path(__file__).parent / "database" / "cluster_manager_schema.sql"
        if not schema_file.exists():
            print("❌ Schema file not found:", schema_file)
            return False
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        print("📄 Schema file loaded")
        
        # Execute schema creation
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
                conn.commit()
        
        print("✅ Database schema initialized successfully")
        
        # Verify tables were created
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'cluster_%'"
        tables = db.execute_query(tables_query, fetch=True)
        
        if tables:
            print(f"✅ Created {len(tables)} cluster tables:")
            for table in tables:
                print(f"  - {table['table_name']}")
        else:
            print("⚠️  No cluster tables found after initialization")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = initialize_schema()
    sys.exit(0 if success else 1)
