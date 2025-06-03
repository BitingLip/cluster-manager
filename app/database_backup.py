"""
Cluster Manager Database Integration
PostgreSQL database operations for cluster state and resource management
"""

import psycopg2
import psycopg2.pool
import psycopg2.extras
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, timedelta
import structlog
import json
import socket
from contextlib import contextmanager

logger = structlog.get_logger(__name__)

class ClusterDatabase:
    """Database manager for cluster state and resource tracking"""
    def __init__(self, config: Dict[str, str]):
        """Initialize database connection pool"""
        self.config = config
        self.connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            db_config = {
                'host': self.config.get('CLUSTER_DB_HOST', 'localhost'),
                'port': int(self.config.get('CLUSTER_DB_PORT', '5432')),
                'database': self.config.get('CLUSTER_DB_NAME', 'cluster_manager'),
                'user': self.config.get('CLUSTER_DB_USER', 'postgres'),
                'password': self.config.get('CLUSTER_DB_PASSWORD', 'password'),
                'minconn': int(self.config.get('CLUSTER_DB_MIN_CONN', '2')),
                'maxconn': int(self.config.get('CLUSTER_DB_MAX_CONN', '10'))
            }
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(**db_config)
            logger.info("Cluster database connection pool initialized", 
                       host=db_config['host'], 
                       database=db_config['database'])
            
        except Exception as e:
            logger.error("Failed to initialize cluster database connection pool", error=str(e))
            raise
      @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if not self.connection_pool:
            raise RuntimeError("Database connection pool not initialized")
            
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        finally:
            if conn and self.connection_pool:
                self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch: bool = False) -> Optional[List[Dict]]:
        """Execute a database query"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query, params or ())
                    
                    if fetch:
                        result = cursor.fetchall()
                        return [dict(row) for row in result] if result else []
                    else:
                        conn.commit()
                        return None
                        
        except Exception as e:
            logger.error("Database query failed", query=query, error=str(e))
            if fetch:
                return []  # Return empty list for fetch queries on error
            raise
    # === NODE MANAGEMENT ===
    
    def register_node(self, node_id: str, hostname: str, ip_address: str, 
                     port: int, node_type: str = 'worker', 
                     capabilities: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> bool:
        """Register a new cluster node"""
        try:
            query = """
                INSERT INTO cluster_nodes 
                (id, hostname, ip_address, port, node_type, status, last_heartbeat, capabilities, metadata)
                VALUES (%s, %s, %s, %s, %s, 'online', NOW(), %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    hostname = EXCLUDED.hostname,
                    ip_address = EXCLUDED.ip_address,
                    port = EXCLUDED.port,
                    node_type = EXCLUDED.node_type,
                    status = 'online',
                    last_heartbeat = NOW(),
                    capabilities = EXCLUDED.capabilities,
                    metadata = EXCLUDED.metadata
            """
            
            self.execute_query(query, (
                node_id, hostname, ip_address, port, node_type,
                json.dumps(capabilities or []),
                json.dumps(metadata or {})
            ))
            
            logger.info("Node registered successfully", 
                       node_id=node_id, hostname=hostname, node_type=node_type)
            return True
            
        except Exception as e:
            logger.error("Failed to register node", node_id=node_id, error=str(e))
            return False
    
    def update_node_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat timestamp"""
        try:
            query = """
                UPDATE cluster_nodes 
                SET last_heartbeat = NOW(), status = 'online'
                WHERE id = %s            """
            # Use explicit connection control to prevent hangs
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (node_id,))
                    conn.commit()  # Explicit commit
            return True
            
        except Exception as e:
            logger.error("Failed to update heartbeat", node_id=node_id, error=str(e))
            return False
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed node information"""
        try:
            query = """
                SELECT * FROM cluster_nodes WHERE id = %s
            """
            result = self.execute_query(query, (node_id,), fetch=True)
            return result[0] if result else None
            
        except Exception as e:
            logger.error("Failed to get node info", node_id=node_id, error=str(e))
            return None
    
    def list_nodes(self, node_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List cluster nodes with optional filtering"""
        try:
            where_conditions = []
            params = []
            
            if node_type:
                where_conditions.append("node_type = %s")
                params.append(node_type)
            
            if status:
                where_conditions.append("status = %s")
                params.append(status)
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            query = f"""
                SELECT * FROM cluster_nodes 
                {where_clause}
                ORDER BY joined_at DESC
            """
            
            return self.execute_query(query, tuple(params), fetch=True) or []
            
        except Exception as e:
            logger.error("Failed to list nodes", error=str(e))
            return []
    
    def mark_node_offline(self, node_id: str) -> bool:
        """Mark a node as offline"""
        try:
            query = """
                UPDATE cluster_nodes 
                SET status = 'offline', left_at = NOW()
                WHERE id = %s
            """
            self.execute_query(query, (node_id,))
            
            logger.info("Node marked offline", node_id=node_id)
            return True
            
        except Exception as e:
            logger.error("Failed to mark node offline", node_id=node_id, error=str(e))
            return False
    
    # === GPU RESOURCE MANAGEMENT ===
    def update_gpu_resources(self, node_id: str, gpu_data: List[Dict[str, Any]]) -> bool:
        """Update GPU resource information for a node"""
        try:
            # Clear existing GPU data for this node
            self.execute_query("DELETE FROM gpu_resources WHERE node_id = %s", (node_id,))
            
            # Insert new GPU data
            insert_query = """
                INSERT INTO gpu_resources 
                (node_id, gpu_index, gpu_name, memory_total_mb, memory_used_mb, 
                 memory_available_mb, utilization_percent, temperature_celsius, 
                 power_usage_watts, driver_version, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for gpu in gpu_data:
                self.execute_query(insert_query, (
                    node_id,
                    gpu['index'],
                    gpu.get('name'),
                    gpu.get('memory_total_mb'),
                    gpu.get('memory_used_mb', 0),
                    gpu.get('memory_available_mb'),
                    gpu.get('utilization_percent', 0.0),
                    gpu.get('temperature_celsius'),
                    gpu.get('power_usage_watts'),
                    gpu.get('driver_version'),
                    json.dumps(gpu.get('metadata', {}))
                ))
            
            logger.debug("GPU resources updated", node_id=node_id, gpu_count=len(gpu_data))
            return True
            
        except Exception as e:
            logger.error("Failed to update GPU resources", node_id=node_id, error=str(e))
            return False
    
    def get_gpu_resources(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get GPU resource information"""
        try:
            if node_id:
                query = "SELECT * FROM gpu_resources WHERE node_id = %s ORDER BY gpu_index"
                params = (node_id,)
            else:
                query = "SELECT * FROM gpu_resources ORDER BY node_id, gpu_index"
                params = ()
            
            return self.execute_query(query, params, fetch=True) or []
            
        except Exception as e:
            logger.error("Failed to get GPU resources", node_id=node_id, error=str(e))
            return []
    
    # === SYSTEM RESOURCE MANAGEMENT ===
    
    def update_system_resources(self, node_id: str, system_data: Dict[str, Any]) -> bool:
        """Update system resource information for a node"""
        try:
            query = """
                INSERT INTO system_resources 
                (node_id, cpu_count, cpu_usage_percent, memory_total_mb, memory_used_mb,
                 memory_available_mb, disk_total_gb, disk_used_gb, disk_available_gb,
                 network_rx_bytes, network_tx_bytes, load_average_1m, load_average_5m, 
                 load_average_15m, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (node_id) DO UPDATE SET
                    cpu_count = EXCLUDED.cpu_count,
                    cpu_usage_percent = EXCLUDED.cpu_usage_percent,
                    memory_total_mb = EXCLUDED.memory_total_mb,
                    memory_used_mb = EXCLUDED.memory_used_mb,
                    memory_available_mb = EXCLUDED.memory_available_mb,
                    disk_total_gb = EXCLUDED.disk_total_gb,
                    disk_used_gb = EXCLUDED.disk_used_gb,
                    disk_available_gb = EXCLUDED.disk_available_gb,
                    network_rx_bytes = EXCLUDED.network_rx_bytes,
                    network_tx_bytes = EXCLUDED.network_tx_bytes,
                    load_average_1m = EXCLUDED.load_average_1m,
                    load_average_5m = EXCLUDED.load_average_5m,
                    load_average_15m = EXCLUDED.load_average_15m,
                    last_updated = NOW(),
                    metadata = EXCLUDED.metadata
            """
            
            self.execute_query(query, (
                node_id,
                system_data.get('cpu_count'),
                system_data.get('cpu_usage_percent'),
                system_data.get('memory_total_mb'),
                system_data.get('memory_used_mb'),
                system_data.get('memory_available_mb'),
                system_data.get('disk_total_gb'),
                system_data.get('disk_used_gb'),
                system_data.get('disk_available_gb'),
                system_data.get('network_rx_bytes'),
                system_data.get('network_tx_bytes'),
                system_data.get('load_average_1m'),
                system_data.get('load_average_5m'),
                system_data.get('load_average_15m'),
                json.dumps(system_data.get('metadata', {}))
            ))
            
            return True
            
        except Exception as e:
            logger.error("Failed to update system resources", node_id=node_id, error=str(e))
            return False
    
    # === CLUSTER HEALTH AND MONITORING ===
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get cluster health summary"""
        try:
            query = "SELECT * FROM cluster_health_summary"
            result = self.execute_query(query, fetch=True)
            return result[0] if result else {}
            
        except Exception as e:
            logger.error("Failed to get cluster health", error=str(e))
            return {}
    
    def get_resource_utilization(self) -> List[Dict[str, Any]]:
        """Get resource utilization across all nodes"""
        try:
            query = "SELECT * FROM resource_utilization ORDER BY node_id"
            return self.execute_query(query, fetch=True) or []
            
        except Exception as e:
            logger.error("Failed to get resource utilization", error=str(e))
            return []
    
    def log_cluster_event(self, event_type: str, severity: str, title: str,
                         description: Optional[str] = None, source_node_id: Optional[str] = None,
                         event_data: Optional[Dict] = None) -> bool:
        """Log a cluster event"""
        try:
            query = """
                INSERT INTO cluster_events 
                (event_type, severity, source_node_id, title, description, event_data)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            self.execute_query(query, (
                event_type, severity, source_node_id, title, description,
                json.dumps(event_data or {})
            ))
            
            return True
            
        except Exception as e:
            logger.error("Failed to log cluster event", error=str(e))
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            query = "SELECT * FROM active_alerts"
            return self.execute_query(query, fetch=True) or []
            
        except Exception as e:
            logger.error("Failed to get active alerts", error=str(e))
            return []
    
    # === CLEANUP AND MAINTENANCE ===
    
    def cleanup_stale_nodes(self, timeout_minutes: int = 10) -> int:
        """Remove nodes that haven't sent heartbeat within timeout"""
        try:
            query = """
                UPDATE cluster_nodes 
                SET status = 'offline', left_at = NOW()
                WHERE last_heartbeat < NOW() - INTERVAL '%s minutes'
                  AND status != 'offline'
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (timeout_minutes,))
                    affected_rows = cursor.rowcount
                    conn.commit()
            
            if affected_rows > 0:
                logger.info("Marked stale nodes as offline", count=affected_rows)
            
            return affected_rows
            
        except Exception as e:
            logger.error("Failed to cleanup stale nodes", error=str(e))
            return 0
    
    def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Cluster database connection pool closed")
