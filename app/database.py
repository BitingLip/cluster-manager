"""
Database module for the cluster manager.
Provides PostgreSQL database connectivity and operations for cluster management.
"""

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ClusterDatabase:
    """Database interface for cluster management operations."""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str, 
                 min_connections: int = 1, max_connections: int = 10):
        """Initialize database connection pool."""
        self.connection_params = {
            'host': host,
            'port': port, 
            'database': database,
            'user': user,
            'password': password
        }
        
        try:
            # Create connection pool
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                **self.connection_params,
                cursor_factory=RealDictCursor
            )
            logger.info(f"Database connection pool created: {min_connections}-{max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool."""
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        try:
            self.connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Failed to return database connection: {e}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                
                # Handle different query types
                if cursor.description:
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                else:
                    conn.commit()
                    return []
                    
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database query failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def update_node_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat timestamp. Uses explicit connection control to prevent hangs."""
        query = """
        UPDATE cluster_nodes 
        SET last_heartbeat = NOW()
        WHERE node_id = %s
        """
        
        # Use explicit connection control instead of execute_query to prevent hangs
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, (node_id,))
                conn.commit()  # Explicit commit
                logger.debug(f"Updated heartbeat for node {node_id}")
                return True
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to update heartbeat for node {node_id}: {e}")
            return False
        finally:
            if conn:
                self.return_connection(conn)
    
    def update_system_resources(self, node_id: str, **kwargs) -> bool:
        """Update system resource metrics for a node - accepts flexible parameters."""
        try:
            # Extract the values we need, calculating percentages if necessary
            cpu_usage = kwargs.get('cpu_percent', kwargs.get('cpu_usage', 0.0))
            
            # Calculate memory usage percentage
            memory_total = kwargs.get('memory_total')
            memory_used = kwargs.get('memory_used')
            if memory_total and memory_used and memory_total > 0:
                memory_usage = (memory_used / memory_total) * 100.0
            else:
                memory_usage = kwargs.get('memory_usage', 0.0)
            
            # Calculate disk usage percentage
            disk_total = kwargs.get('disk_total')
            disk_used = kwargs.get('disk_used')
            if disk_total and disk_used and disk_total > 0:
                disk_usage = (disk_used / disk_total) * 100.0
            else:
                disk_usage = kwargs.get('disk_usage', 0.0)
            
            gpu_usage = kwargs.get('gpu_usage')
            
            query = """
            INSERT INTO system_resources (node_id, cpu_usage, memory_usage, disk_usage, gpu_usage, timestamp)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (node_id) DO UPDATE SET
                cpu_usage = EXCLUDED.cpu_usage,
                memory_usage = EXCLUDED.memory_usage,
                disk_usage = EXCLUDED.disk_usage,
                gpu_usage = EXCLUDED.gpu_usage,
                timestamp = EXCLUDED.timestamp
            """
            
            self.execute_query(query, (node_id, cpu_usage, memory_usage, disk_usage, gpu_usage))
            logger.debug(f"Updated system resources for node {node_id}: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%, Disk={disk_usage:.1f}%")
            return True
        except Exception as e:
            logger.error(f"Failed to update system resources for node {node_id}: {e}")
            return False
    
    def register_node(self, node_id: str, node_type: str, host: Optional[str] = None, port: Optional[int] = None, 
                     hostname: Optional[str] = None, ip_address: Optional[str] = None, capabilities: Optional[List[str]] = None,
                     metadata: Optional[Dict] = None) -> bool:
        """Register a new node in the cluster."""
        # Use hostname and ip_address if provided, otherwise use host
        actual_host = hostname or host or 'localhost'
        actual_port = port or 8000
        
        # Combine capabilities into metadata
        if metadata is None:
            metadata = {}
        if capabilities:
            metadata['capabilities'] = capabilities
        if ip_address:
            metadata['ip_address'] = ip_address
            
        query = """
        INSERT INTO cluster_nodes (node_id, node_type, host, port, status, metadata, last_heartbeat)
        VALUES (%s, %s, %s, %s, 'active', %s, NOW())
        ON CONFLICT (node_id) DO UPDATE SET
            node_type = EXCLUDED.node_type,
            host = EXCLUDED.host,
            port = EXCLUDED.port,
            status = 'active',
            metadata = EXCLUDED.metadata,
            last_heartbeat = NOW()
        """
        
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            self.execute_query(query, (node_id, node_type, actual_host, actual_port, metadata_json))
            logger.info(f"Registered node {node_id} of type {node_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return False
    
    def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get all active nodes in the cluster."""
        query = """
        SELECT node_id, node_type, host, port, status, metadata, last_heartbeat
        FROM cluster_nodes 
        WHERE status = 'active'
        AND last_heartbeat > NOW() - INTERVAL '5 minutes'
        ORDER BY last_heartbeat DESC
        """
        
        try:
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []
    
    def get_node_resources(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get latest system resources for a specific node."""
        query = """
        SELECT cpu_usage, memory_usage, disk_usage, gpu_usage, timestamp
        FROM system_resources
        WHERE node_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        try:
            results = self.execute_query(query, (node_id,))
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Failed to get resources for node {node_id}: {e}")
            return None
    
    def log_cluster_event(self, event_type: str, event_data: Dict[str, Any], 
                         severity: str = 'info', node_id: Optional[str] = None) -> bool:
        """Log a cluster event."""
        # Since we don't have cluster_events table in our simplified schema,
        # just log to the application logger for now
        logger.info(f"Cluster Event [{severity.upper()}] {event_type}: {event_data}")
        return True
    
    def mark_node_offline(self, node_id: str) -> bool:
        """Mark a node as offline."""
        query = """
        UPDATE cluster_nodes 
        SET status = 'offline'
        WHERE node_id = %s
        """
        
        try:
            self.execute_query(query, (node_id,))
            logger.info(f"Marked node {node_id} as offline")
            return True
        except Exception as e:
            logger.error(f"Failed to mark node {node_id} as offline: {e}")
            return False
    
    def cleanup_stale_nodes(self, timeout_minutes: int = 10) -> int:
        """Cleanup stale nodes that haven't sent heartbeat in timeout_minutes."""
        query = """
        UPDATE cluster_nodes 
        SET status = 'offline'
        WHERE status = 'active'
        AND last_heartbeat < NOW() - INTERVAL '%s minutes'
        RETURNING node_id
        """
        
        try:
            results = self.execute_query(query, (timeout_minutes,))
            count = len(results)
            if count > 0:
                node_ids = [row['node_id'] for row in results]
                logger.info(f"Marked {count} stale nodes as offline: {node_ids}")
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup stale nodes: {e}")
            return 0
    
    def update_gpu_resources(self, node_id: str, gpu_data: Dict[str, Any]) -> bool:
        """Update GPU resource information for a node"""
        query = """
        INSERT INTO gpu_resources (node_id, gpu_index, gpu_name, vendor, memory_total_mb, 
                                  memory_used_mb, utilization_percent, temperature_celsius, 
                                  ai_capabilities, metadata, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (node_id, gpu_index) 
        DO UPDATE SET 
            memory_used_mb = EXCLUDED.memory_used_mb,
            utilization_percent = EXCLUDED.utilization_percent,
            temperature_celsius = EXCLUDED.temperature_celsius,
            updated_at = NOW()
        """
        
        try:
            params = (
                node_id,
                gpu_data.get('index', 0),
                gpu_data.get('name', 'Unknown'),
                gpu_data.get('vendor', 'Unknown'),
                gpu_data.get('memory_total_mb', 0),
                gpu_data.get('memory_used_mb', 0),
                gpu_data.get('utilization_percent', 0),
                gpu_data.get('temperature_celsius'),
                json.dumps(gpu_data.get('ai_capabilities', {})),
                json.dumps(gpu_data.get('metadata', {}))
            )
            
            self.execute_query(query, params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update GPU resources for node {node_id}: {e}")
            return False

    def get_cluster_health(self) -> Dict[str, Any]:
        """Get overall cluster health status"""
        try:
            # Count nodes by status
            query = """
            SELECT status, COUNT(*) as count 
            FROM cluster_nodes 
            GROUP BY status
            """
            results = self.execute_query(query)
            
            status_counts = {row['status']: row['count'] for row in results}
            total_nodes = sum(status_counts.values())
            
            # Calculate health percentage
            healthy_nodes = status_counts.get('online', 0)
            health_percentage = (healthy_nodes / total_nodes * 100) if total_nodes > 0 else 0
            
            return {
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_nodes,
                'health_percentage': health_percentage,
                'status_breakdown': status_counts,
                'cluster_status': 'healthy' if health_percentage > 80 else 'degraded' if health_percentage > 50 else 'critical'
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster health: {e}")
            return {'cluster_status': 'unknown', 'error': str(e)}

    def list_nodes(self) -> List[Dict[str, Any]]:
        """List all nodes in the cluster"""
        query = """
        SELECT node_id, hostname, ip_address, port, node_type, status, 
               capabilities, metadata, created_at, last_heartbeat
        FROM cluster_nodes
        ORDER BY created_at DESC
        """
        
        try:
            results = self.execute_query(query)
            # Convert capabilities and metadata from JSON
            for result in results:
                if result.get('capabilities'):
                    result['capabilities'] = json.loads(result['capabilities'])
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
            return results
            
        except Exception as e:
            logger.error(f"Failed to list nodes: {e}")
            return []

    def get_resource_utilization(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get resource utilization for all nodes or specific node"""
        if node_id:
            query = """
            SELECT * FROM system_resources 
            WHERE node_id = %s
            ORDER BY timestamp DESC LIMIT 1
            """
            params = (node_id,)
        else:
            query = """
            SELECT sr.*, cn.hostname 
            FROM system_resources sr
            JOIN cluster_nodes cn ON sr.node_id = cn.node_id
            WHERE sr.timestamp = (
                SELECT MAX(timestamp) 
                FROM system_resources sr2 
                WHERE sr2.node_id = sr.node_id
            )
            ORDER BY sr.timestamp DESC
            """
            params = None
        
        try:
            results = self.execute_query(query, params)
            return results
            
        except Exception as e:
            logger.error(f"Failed to get resource utilization: {e}")
            return []

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts for the cluster"""
        query = """
        SELECT * FROM cluster_alerts 
        WHERE status = 'active'
        ORDER BY created_at DESC
        LIMIT 50
        """
        
        try:
            results = self.execute_query(query)
            # Parse metadata JSON
            for result in results:
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
            return results
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node"""
        query = """
        SELECT * FROM cluster_nodes 
        WHERE node_id = %s
        """
        
        try:
            results = self.execute_query(query, (node_id,))
            if results:
                node_info = results[0]
                # Parse JSON fields
                if node_info.get('capabilities'):
                    node_info['capabilities'] = json.loads(node_info['capabilities'])
                if node_info.get('metadata'):
                    node_info['metadata'] = json.loads(node_info['metadata'])
                return node_info
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node info for {node_id}: {e}")
            return None
    
    def initialize_schema(self) -> bool:
        """Initialize database schema for cluster management."""
        schema_queries = [
            """
            CREATE TABLE IF NOT EXISTS cluster_nodes (
                node_id VARCHAR(255) PRIMARY KEY,
                node_type VARCHAR(50) NOT NULL,
                host VARCHAR(255) NOT NULL,
                port INTEGER NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                metadata JSONB,
                last_heartbeat TIMESTAMP DEFAULT NOW(),
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_resources (
                node_id VARCHAR(255) PRIMARY KEY,
                cpu_usage DECIMAL(5,2),
                memory_usage DECIMAL(5,2),
                disk_usage DECIMAL(5,2),
                gpu_usage DECIMAL(5,2),
                timestamp TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (node_id) REFERENCES cluster_nodes(node_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_status ON cluster_nodes(status);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_heartbeat ON cluster_nodes(last_heartbeat);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_resources_timestamp ON system_resources(timestamp);
            """
        ]
        
        try:
            for query in schema_queries:
                self.execute_query(query)
            logger.info("Database schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            return False
    
    def close(self):
        """Close all database connections."""
        try:
            if hasattr(self, 'connection_pool'):
                self.connection_pool.closeall()
                logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
