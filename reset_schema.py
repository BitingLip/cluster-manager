#!/usr/bin/env python3
"""
Check and reset database schema
"""

import psycopg2
from psycopg2.extras import RealDictCursor

def check_and_reset_schema():
    """Check current schema and reset if needed"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='bitinglip_cluster',
            user='cluster_manager',
            password='cluster_manager_2025!',
            cursor_factory=RealDictCursor
        )
        
        cursor = conn.cursor()
        
        # Check existing tables
        cursor.execute("""
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            ORDER BY table_name, ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        if columns:
            print("Existing schema:")
            current_table = None
            for col in columns:
                if col['table_name'] != current_table:
                    current_table = col['table_name']
                    print(f"\nTable: {current_table}")
                print(f"  - {col['column_name']}: {col['data_type']}")
            
            # Drop existing tables
            print("\nDropping existing tables...")
            cursor.execute("DROP TABLE IF EXISTS system_resources CASCADE")
            cursor.execute("DROP TABLE IF EXISTS cluster_nodes CASCADE")
            conn.commit()
            print("✅ Tables dropped successfully")
        else:
            print("No existing tables found")
        
        # Create new schema
        print("\nCreating new schema...")
        
        # Create cluster_nodes table
        cursor.execute("""
            CREATE TABLE cluster_nodes (
                node_id VARCHAR(255) PRIMARY KEY,
                node_type VARCHAR(50) NOT NULL,
                host VARCHAR(255) NOT NULL,
                port INTEGER NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                metadata JSONB,
                last_heartbeat TIMESTAMP DEFAULT NOW(),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create system_resources table
        cursor.execute("""
            CREATE TABLE system_resources (
                node_id VARCHAR(255) PRIMARY KEY,
                cpu_usage DECIMAL(5,2),
                memory_usage DECIMAL(5,2),
                disk_usage DECIMAL(5,2),
                gpu_usage DECIMAL(5,2),
                timestamp TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (node_id) REFERENCES cluster_nodes(node_id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX idx_nodes_status ON cluster_nodes(status)")
        cursor.execute("CREATE INDEX idx_nodes_heartbeat ON cluster_nodes(last_heartbeat)")
        cursor.execute("CREATE INDEX idx_resources_timestamp ON system_resources(timestamp)")
        
        conn.commit()
        print("✅ New schema created successfully")
        
        # Verify schema
        cursor.execute("""
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            ORDER BY table_name, ordinal_position
        """)
        
        columns = cursor.fetchall()
        print("\nNew schema verification:")
        current_table = None
        for col in columns:
            if col['table_name'] != current_table:
                current_table = col['table_name']
                print(f"\nTable: {current_table}")
            print(f"  - {col['column_name']}: {col['data_type']}")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 Schema reset completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Schema reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_and_reset_schema()
