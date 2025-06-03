#!/usr/bin/env python3
"""
Migrate cluster manager database schema
"""

import sys
import os
from pathlib import Path

# Add cluster manager app to path
cluster_app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(cluster_app_path))

from database import ClusterDatabase

def migrate_schema():
    """Migrate the database schema to match our requirements"""
    
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
    
    print("Migrating cluster manager database schema...")
    
    try:
        # Initialize database connection
        db = ClusterDatabase(config)
        print("✅ Database connection established")
        
        # Migration queries
        migration_queries = [
            # Add missing columns to cluster_nodes
            """
            ALTER TABLE cluster_nodes 
            ADD COLUMN IF NOT EXISTS port INTEGER DEFAULT 8080,
            ADD COLUMN IF NOT EXISTS node_type TEXT DEFAULT 'worker',
            ADD COLUMN IF NOT EXISTS left_at TIMESTAMP,
            ADD COLUMN IF NOT EXISTS version TEXT,
            ADD COLUMN IF NOT EXISTS capabilities JSONB DEFAULT '[]'::jsonb;
            """,
            
            # Create GPU resources table
            """
            CREATE TABLE IF NOT EXISTS gpu_resources (
                id SERIAL PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
                gpu_index INTEGER NOT NULL,
                gpu_name TEXT,
                memory_total INTEGER,
                memory_used INTEGER DEFAULT 0,
                memory_free INTEGER,
                utilization REAL DEFAULT 0.0,
                temperature REAL,
                power_usage REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(node_id, gpu_index)
            );
            """,
            
            # Create system resources table
            """
            CREATE TABLE IF NOT EXISTS system_resources (
                id SERIAL PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
                cpu_percent REAL DEFAULT 0.0,
                memory_total BIGINT,
                memory_used BIGINT DEFAULT 0,
                memory_free BIGINT,
                disk_total BIGINT,
                disk_used BIGINT DEFAULT 0,
                disk_free BIGINT,
                network_sent BIGINT DEFAULT 0,
                network_recv BIGINT DEFAULT 0,
                load_average REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(node_id)
            );
            """,
            
            # Create resource allocations table
            """
            CREATE TABLE IF NOT EXISTS resource_allocations (
                id SERIAL PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                allocated_to TEXT,
                allocation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                metadata JSONB DEFAULT '{}'::jsonb
            );
            """,
            
            # Create cluster events table
            """
            CREATE TABLE IF NOT EXISTS cluster_events (
                id SERIAL PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_data JSONB DEFAULT '{}'::jsonb,
                severity TEXT DEFAULT 'info',
                node_id TEXT REFERENCES cluster_nodes(id) ON DELETE CASCADE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            
            # Create load balancing metrics table
            """
            CREATE TABLE IF NOT EXISTS load_balancing_metrics (
                id SERIAL PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
                metric_type TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            
            # Create cluster config table
            """
            CREATE TABLE IF NOT EXISTS cluster_config (
                id SERIAL PRIMARY KEY,
                config_key TEXT UNIQUE NOT NULL,
                config_value JSONB NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            
            # Create network topology table
            """
            CREATE TABLE IF NOT EXISTS network_topology (
                id SERIAL PRIMARY KEY,
                source_node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
                target_node_id TEXT NOT NULL REFERENCES cluster_nodes(id) ON DELETE CASCADE,
                connection_type TEXT DEFAULT 'tcp',
                latency_ms REAL,
                bandwidth_mbps REAL,
                last_measured TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_node_id, target_node_id)
            );
            """
        ]
        
        # Execute migration queries
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                for i, query in enumerate(migration_queries, 1):
                    print(f"📝 Executing migration step {i}/{len(migration_queries)}")
                    cursor.execute(query)
                    conn.commit()
        
        print("✅ Database migration completed successfully")
        
        # Verify tables were created
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        tables = db.execute_query(tables_query, fetch=True)
        
        print(f"\n📊 Database now contains {len(tables)} tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        # Check cluster tables specifically
        cluster_tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'cluster_%' OR table_name LIKE '%resources%' OR table_name LIKE '%allocations%' OR table_name LIKE '%events%' OR table_name LIKE '%metrics%' OR table_name LIKE '%topology%' ORDER BY table_name"
        cluster_tables = db.execute_query(cluster_tables_query, fetch=True)
        
        print(f"\n🎯 Cluster-related tables ({len(cluster_tables)}):")
        for table in cluster_tables:
            print(f"  - {table['table_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_schema()
    sys.exit(0 if success else 1)
