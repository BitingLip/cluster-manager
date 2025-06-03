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
                'database': self.config.get('CLUSTER_DB_NAME', 'bitinglip_cluster'),
                'user': self.config.get('CLUSTER_DB_USER', 'cluster_manager'),
                'password': self.config.get('CLUSTER_DB_PASSWORD', 'cluster_manager_2025!'),
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
                WHERE id = %s
            """
            self.execute_query(query, (node_id,))
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
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            query = f"SELECT * FROM cluster_nodes {where_clause} ORDER BY last_heartbeat DESC"
            
            result = self.execute_query(query, tuple(params), fetch=True)
            return result or []
            
        except Exception as e:
            logger.error("Failed to list nodes", error=str(e))
            return []
    
    def mark_node_offline(self, node_id: str) -> bool:
        """Mark a node as offline"""
        try:
            query = """
                UPDATE cluster_nodes 
                SET status = 'offline'
                WHERE id = %s
            """
            self.execute_query(query, (node_id,))
            return True
            
        except Exception as e:
            logger.error("Failed to mark node offline", node_id=node_id, error=str(e))
            return False
    
    # === RESOURCE MANAGEMENT ===
    
    def update_gpu_resources(self, node_id: str, gpu_data: List[Dict]) -> bool:
        """Update GPU resource information for a node"""
        try:
            # Clear existing GPU data for this node
            delete_query = "DELETE FROM gpu_resources WHERE node_id = %s"
            self.execute_query(delete_query, (node_id,))
            
            # Insert new GPU data
            insert_query = """
                INSERT INTO gpu_resources 
                (node_id, gpu_index, gpu_name, memory_total, memory_used, memory_free, utilization, temperature, power_usage)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for gpu in gpu_data:
                self.execute_query(insert_query, (
                    node_id,
                    gpu.get('index'),
                    gpu.get('name'),
                    gpu.get('memory_total'),
                    gpu.get('memory_used'),
                    gpu.get('memory_free'),
                    gpu.get('utilization'),
                    gpu.get('temperature'),
                    gpu.get('power_usage')
                ))
            
            return True
            
        except Exception as e:
            logger.error("Failed to update GPU resources", node_id=node_id, error=str(e))
            return False
    
    def update_system_resources(self, node_id: str, cpu_percent: float, memory_total: int, 
                               memory_used: int, disk_total: int, disk_used: int, 
                               network_sent: int, network_recv: int, load_avg: Optional[float] = None) -> bool:
        """Update system resource information for a node"""
        try:
            query = """
                INSERT INTO system_resources 
                (node_id, cpu_percent, memory_total, memory_used, memory_free, 
                 disk_total, disk_used, disk_free, network_sent, network_recv, load_average, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (node_id) DO UPDATE SET
                    cpu_percent = EXCLUDED.cpu_percent,
                    memory_total = EXCLUDED.memory_total,
                    memory_used = EXCLUDED.memory_used,
                    memory_free = EXCLUDED.memory_free,
                    disk_total = EXCLUDED.disk_total,
                    disk_used = EXCLUDED.disk_used,
                    disk_free = EXCLUDED.disk_free,
                    network_sent = EXCLUDED.network_sent,
                    network_recv = EXCLUDED.network_recv,
                    load_average = EXCLUDED.load_average,
                    timestamp = NOW()
            """
            
            memory_free = memory_total - memory_used
            disk_free = disk_total - disk_used
            
            self.execute_query(query, (
                node_id, cpu_percent, memory_total, memory_used, memory_free,
                disk_total, disk_used, disk_free, network_sent, network_recv, load_avg
            ))
            
            return True
            
        except Exception as e:
            logger.error("Failed to update system resources", node_id=node_id, error=str(e))
            return False
    
    def get_resource_utilization(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get resource utilization for nodes"""
        try:
            if node_id:
                query = """
                    SELECT cn.id, cn.hostname, cn.status,
                           sr.cpu_percent, sr.memory_used, sr.memory_total,
                           sr.disk_used, sr.disk_total, sr.load_average,
                           COUNT(gr.gpu_index) as gpu_count,
                           AVG(gr.utilization) as avg_gpu_utilization
                    FROM cluster_nodes cn
                    LEFT JOIN system_resources sr ON cn.id = sr.node_id
                    LEFT JOIN gpu_resources gr ON cn.id = gr.node_id
                    WHERE cn.id = %s
                    GROUP BY cn.id, cn.hostname, cn.status, sr.cpu_percent, 
                             sr.memory_used, sr.memory_total, sr.disk_used, 
                             sr.disk_total, sr.load_average
                """
                result = self.execute_query(query, (node_id,), fetch=True)
            else:
                query = """
                    SELECT cn.id, cn.hostname, cn.status,
                           sr.cpu_percent, sr.memory_used, sr.memory_total,
                           sr.disk_used, sr.disk_total, sr.load_average,
                           COUNT(gr.gpu_index) as gpu_count,
                           AVG(gr.utilization) as avg_gpu_utilization
                    FROM cluster_nodes cn
                    LEFT JOIN system_resources sr ON cn.id = sr.node_id
                    LEFT JOIN gpu_resources gr ON cn.id = gr.node_id
                    GROUP BY cn.id, cn.hostname, cn.status, sr.cpu_percent, 
                             sr.memory_used, sr.memory_total, sr.disk_used, 
                             sr.disk_total, sr.load_average
                    ORDER BY cn.hostname
                """
                result = self.execute_query(query, fetch=True)
            
            return result or []
            
        except Exception as e:
            logger.error("Failed to get resource utilization", error=str(e))
            return []
    
    # === CLUSTER MONITORING ===
    
    def log_cluster_event(self, event_type: str, event_data: Dict, severity: str = 'info', 
                         node_id: Optional[str] = None) -> bool:
        """Log a cluster event"""
        try:
            query = """
                INSERT INTO cluster_events (event_type, event_data, severity, node_id, timestamp)
                VALUES (%s, %s, %s, %s, NOW())
            """
            
            self.execute_query(query, (
                event_type, json.dumps(event_data), severity, node_id
            ))
            
            return True
            
        except Exception as e:
            logger.error("Failed to log cluster event", error=str(e))
            return False
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get overall cluster health status"""
        try:
            # Get node counts by status
            node_status_query = """
                SELECT status, COUNT(*) as count
                FROM cluster_nodes
                GROUP BY status
            """
            node_status = self.execute_query(node_status_query, fetch=True)
            
            # Get total resource utilization
            resource_query = """
                SELECT 
                    COUNT(DISTINCT cn.id) as total_nodes,
                    AVG(sr.cpu_percent) as avg_cpu_usage,
                    SUM(sr.memory_used)::BIGINT as total_memory_used,
                    SUM(sr.memory_total)::BIGINT as total_memory,
                    COUNT(gr.gpu_index) as total_gpus,
                    AVG(gr.utilization) as avg_gpu_utilization
                FROM cluster_nodes cn
                LEFT JOIN system_resources sr ON cn.id = sr.node_id
                LEFT JOIN gpu_resources gr ON cn.id = gr.node_id
                WHERE cn.status = 'online'
            """
            resource_data = self.execute_query(resource_query, fetch=True)
            
            # Get recent alerts
            alerts_query = """
                SELECT COUNT(*) as alert_count
                FROM cluster_events
                WHERE severity IN ('warning', 'error', 'critical')
                AND timestamp > NOW() - INTERVAL '1 hour'
            """
            alerts_data = self.execute_query(alerts_query, fetch=True)
              health_data = {
                'node_status': {row['status']: row['count'] for row in (node_status or [])},
                'total_nodes': resource_data[0]['total_nodes'] if resource_data and resource_data[0] else 0,
                'avg_cpu_usage': resource_data[0]['avg_cpu_usage'] if resource_data and resource_data[0] else 0,
                'memory_usage_gb': (resource_data[0]['total_memory_used'] or 0) / (1024**3) if resource_data and resource_data[0] else 0,
                'total_memory_gb': (resource_data[0]['total_memory'] or 0) / (1024**3) if resource_data and resource_data[0] else 0,
                'total_gpus': resource_data[0]['total_gpus'] if resource_data and resource_data[0] else 0,
                'avg_gpu_utilization': resource_data[0]['avg_gpu_utilization'] if resource_data and resource_data[0] else 0,
                'recent_alerts': alerts_data[0]['alert_count'] if alerts_data and alerts_data[0] else 0
            }
            
            return health_data
            
        except Exception as e:
            logger.error("Failed to get cluster health", error=str(e))
            return {}
    
    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent cluster alerts"""
        try:
            query = """
                SELECT event_type, event_data, severity, node_id, timestamp
                FROM cluster_events
                WHERE severity IN ('warning', 'error', 'critical')
                ORDER BY timestamp DESC
                LIMIT %s
            """
            
            result = self.execute_query(query, (limit,), fetch=True)
            return result or []
            
        except Exception as e:
            logger.error("Failed to get active alerts", error=str(e))
            return []
    
    # === MAINTENANCE ===
    
    def cleanup_stale_nodes(self, timeout_minutes: int = 5) -> int:
        """Mark nodes as offline if they haven't sent heartbeat recently"""
        try:
            query = """
                UPDATE cluster_nodes 
                SET status = 'offline'
                WHERE last_heartbeat < NOW() - INTERVAL '%s minutes'
                AND status = 'online'
            """
            
            # Get count of nodes that will be marked offline
            count_query = """
                SELECT COUNT(*) as count
                FROM cluster_nodes 
                WHERE last_heartbeat < NOW() - INTERVAL '%s minutes'
                AND status = 'online'
            """
            
            count_result = self.execute_query(count_query % timeout_minutes, fetch=True)
            stale_count = count_result[0]['count'] if count_result else 0
            
            # Mark them offline
            self.execute_query(query % timeout_minutes)
            
            if stale_count > 0:
                logger.info("Marked stale nodes offline", count=stale_count, timeout_minutes=timeout_minutes)
            
            return stale_count
            
        except Exception as e:
            logger.error("Failed to cleanup stale nodes", error=str(e))
            return 0
    
    def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Cluster database connection pool closed")
