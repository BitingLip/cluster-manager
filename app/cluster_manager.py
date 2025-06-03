"""
Cluster Manager Main Application
Orchestrates GPU cluster operations with database persistence
"""

import os
import sys
import socket
import threading
import time
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog

# Add project root to path for config imports
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from current app directory
app_path = Path(__file__).parent
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

from simple_config import ClusterManagerSettings
from database import ClusterDatabase

logger = structlog.get_logger(__name__)

class ClusterManager:
    """Main cluster management orchestrator with database integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize cluster manager"""
        # Load configuration
        self.settings = ClusterManagerSettings()        # Initialize database
        try:
            self.database = ClusterDatabase(
                host=self.settings.get_config_value('CLUSTER_DB_HOST', 'localhost'),
                port=int(self.settings.get_config_value('CLUSTER_DB_PORT', '5432')),
                database=self.settings.get_config_value('CLUSTER_DB_NAME', 'bitinglip_cluster'),
                user=self.settings.get_config_value('CLUSTER_DB_USER', 'postgres'),
                password=self.settings.get_config_value('CLUSTER_DB_PASSWORD', 'password'),
                min_connections=1,
                max_connections=10
            )
            logger.info("Cluster manager database initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize cluster database", error=str(e))
            raise
        
        # Node information
        self.node_id = self._generate_node_id()
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()
        self.port = self.settings.port
        
        # Threading control
        self._shutdown_event = threading.Event()
        self._monitor_thread = None
        
        logger.info("Cluster manager initialized", 
                   node_id=self.node_id, 
                   hostname=self.hostname,
                   ip_address=self.ip_address,
                   port=self.port)
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cluster_manager_{hostname}_{timestamp}"
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def start(self) -> bool:
        """Start cluster manager services"""
        try:
            # Register this node as cluster manager
            success = self.database.register_node(
                node_id=self.node_id,
                hostname=self.hostname,
                ip_address=self.ip_address,
                port=self.port,
                node_type='master',
                capabilities=['cluster_management', 'resource_monitoring', 'load_balancing'],
                metadata={
                    'startup_time': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            )
            
            if not success:
                logger.error("Failed to register cluster manager node")
                return False
              # Log startup event
            self.database.log_cluster_event(
                event_type='cluster_manager_start',
                event_data={
                    'startup_time': datetime.now().isoformat(),
                    'node_id': self.node_id,
                    'title': 'Cluster Manager Started',
                    'description': f'Cluster manager started on {self.hostname}:{self.port}'
                },
                severity='info',
                node_id=self.node_id
            )
            
            # Start monitoring thread
            self._monitor_thread = threading.Thread(target=self._monitoring_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            
            logger.info("Cluster manager started successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to start cluster manager", error=str(e))
            return False
    
    def stop(self):
        """Stop cluster manager services"""
        try:
            logger.info("Stopping cluster manager...")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Wait for monitoring thread
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)
            
            # Mark node as offline
            self.database.mark_node_offline(self.node_id)
              # Log shutdown event
            self.database.log_cluster_event(
                event_type='cluster_manager_stop',
                event_data={
                    'shutdown_time': datetime.now().isoformat(),
                    'node_id': self.node_id,
                    'title': 'Cluster Manager Stopped',
                    'description': f'Cluster manager stopped on {self.hostname}'
                },
                severity='info',
                node_id=self.node_id
            )            
            # Close database connections
            self.database.close()
            
            logger.info("Cluster manager stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping cluster manager", error=str(e))
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring loop")
        
        while not self._shutdown_event.is_set():
            try:
                logger.info("Monitoring loop iteration starting")
                
                # Update heartbeat
                logger.info("Updating node heartbeat")
                self.database.update_node_heartbeat(self.node_id)
                
                # Collect and update system resources
                logger.info("Updating system resources")
                self._update_system_resources()
                logger.info("System resources updated successfully")
                
                # Collect and update GPU resources (if available)
                logger.info("Updating GPU resources")
                self._update_gpu_resources()
                logger.info("GPU resources updated successfully")
                
                # Cleanup stale nodes
                logger.info("Cleaning up stale nodes")
                stale_count = self.database.cleanup_stale_nodes(timeout_minutes=10)
                if stale_count > 0:
                    logger.info("Cleaned up stale nodes", count=stale_count)
                
                logger.info("Monitoring loop iteration completed, sleeping for 30 seconds")
                # Sleep for monitoring interval
                self._shutdown_event.wait(30)  # 30-second monitoring interval
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                import traceback
                logger.error("Full traceback", traceback=traceback.format_exc())
                self._shutdown_event.wait(10)  # Wait before retrying
    
    def _update_system_resources(self):
        """Update system resource information"""
        try:
            # Get CPU information
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Get memory information
            memory = psutil.virtual_memory()
            memory_total_mb = memory.total // (1024 * 1024)
            memory_used_mb = memory.used // (1024 * 1024)
            memory_available_mb = memory.available // (1024 * 1024)
              # Get disk information
            try:
                # Use different disk paths for different platforms
                if sys.platform == 'win32':
                    disk = psutil.disk_usage('C:')
                else:
                    disk = psutil.disk_usage('/')
                disk_total_gb = disk.total // (1024 * 1024 * 1024)
                disk_used_gb = disk.used // (1024 * 1024 * 1024)
                disk_available_gb = disk.free // (1024 * 1024 * 1024)
            except Exception as e:
                logger.warning("Failed to get disk usage", error=str(e))
                disk_total_gb = 0
                disk_used_gb = 0
                disk_available_gb = 0
            
            # Get network information
            net_io = psutil.net_io_counters()
            network_rx_bytes = net_io.bytes_recv
            network_tx_bytes = net_io.bytes_sent
            
            # Get load average (Unix only)
            try:
                load_avg = os.getloadavg()
                load_average_1m = load_avg[0]
                load_average_5m = load_avg[1]
                load_average_15m = load_avg[2]
            except (AttributeError, OSError):
                # Windows doesn't have load average
                load_average_1m = cpu_usage / 100.0
                load_average_5m = cpu_usage / 100.0
                load_average_15m = cpu_usage / 100.0
            
            system_data = {
                'cpu_count': cpu_count,
                'cpu_usage_percent': cpu_usage,
                'memory_total_mb': memory_total_mb,
                'memory_used_mb': memory_used_mb,
                'memory_available_mb': memory_available_mb,
                'disk_total_gb': disk_total_gb,
                'disk_used_gb': disk_used_gb,
                'disk_available_gb': disk_available_gb,
                'network_rx_bytes': network_rx_bytes,
                'network_tx_bytes': network_tx_bytes,
                'load_average_1m': load_average_1m,
                'load_average_5m': load_average_5m,
                'load_average_15m': load_average_15m,
                'metadata': {
                    'collection_time': datetime.now().isoformat(),
                    'psutil_version': psutil.__version__
                }            }
            
            # Update system resources in database
            self.database.update_system_resources(
                node_id=self.node_id,
                cpu_percent=cpu_usage,
                memory_total=int(memory_total_mb * 1024 * 1024),  # Convert to bytes
                memory_used=int(memory_used_mb * 1024 * 1024),    # Convert to bytes  
                disk_total=int(disk_total_gb * 1024 * 1024 * 1024),  # Convert to bytes
                disk_used=int(disk_used_gb * 1024 * 1024 * 1024),    # Convert to bytes
                network_sent=network_tx_bytes,
                network_recv=network_rx_bytes,
                load_avg=load_average_1m
            )
            
        except Exception as e:
            logger.error("Failed to update system resources", error=str(e))
    
    def _update_gpu_resources(self):
        """Update GPU resource information"""
        try:
            gpu_data = []
            
            # Try to get GPU information using various methods
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get GPU name
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total_mb = mem_info.total // (1024 * 1024)
                    memory_used_mb = mem_info.used // (1024 * 1024)
                    memory_available_mb = (mem_info.total - mem_info.used) // (1024 * 1024)
                    
                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization_percent = util.gpu
                    
                    # Get temperature
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temperature = None
                    
                    # Get power usage
                    try:
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power_usage = None
                    
                    gpu_info = {
                        'index': i,
                        'name': name,
                        'memory_total_mb': memory_total_mb,
                        'memory_used_mb': memory_used_mb,
                        'memory_available_mb': memory_available_mb,
                        'utilization_percent': utilization_percent,
                        'temperature_celsius': temperature,
                        'power_usage_watts': power_usage,
                        'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                        'metadata': {
                            'collection_time': datetime.now().isoformat(),
                            'library': 'pynvml'
                        }
                    }
                    
                    gpu_data.append(gpu_info)
                
                logger.debug("Collected GPU information using pynvml", gpu_count=len(gpu_data))
                
            except ImportError:
                logger.debug("pynvml not available, skipping GPU monitoring")
            except Exception as e:
                logger.warning("Failed to collect GPU information", error=str(e))
            
            # Update database if we have GPU data
            if gpu_data:
                self.database.update_gpu_resources(self.node_id, gpu_data)
            
        except Exception as e:
            logger.error("Failed to update GPU resources", error=str(e))
    
    # === PUBLIC API METHODS ===
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        try:
            health = self.database.get_cluster_health()
            nodes = self.database.list_nodes()
            utilization = self.database.get_resource_utilization()
            alerts = self.database.get_active_alerts()
            
            return {
                'health': health,
                'nodes': nodes,
                'resource_utilization': utilization,
                'active_alerts': alerts,
                'manager_node_id': self.node_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get cluster status", error=str(e))
            return {}
    
    def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node"""
        try:
            node_info = self.database.get_node_info(node_id)
            if not node_info:
                return None
            
            # Get resource utilization for this node
            resources = self.database.get_resource_utilization(node_id)
            
            return {
                'node_info': node_info,
                'resources': resources,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error("Failed to get node details", node_id=node_id, error=str(e))
            return None
    
    def register_worker_node(self, node_id: str, hostname: str, ip_address: str,
                           port: int, capabilities: Optional[List[str]] = None,
                           metadata: Optional[Dict] = None) -> bool:
        """Register a new worker node"""
        try:
            success = self.database.register_node(
                node_id=node_id,
                hostname=hostname,
                ip_address=ip_address,
                port=port,
                node_type='worker',
                capabilities=capabilities or [],
                metadata=metadata or {}
            )
            
            if success:
                self.database.log_cluster_event(
                    event_type='worker_join',
                    event_data={
                        'node_id': node_id,
                        'hostname': hostname,
                        'ip_address': ip_address,
                        'port': port,
                        'title': 'Worker Node Joined',
                        'description': f'Worker node {hostname} joined the cluster'
                    },
                    severity='info',
                    node_id=node_id
                )
            
            return success
            
        except Exception as e:
            logger.error("Failed to register worker node", node_id=node_id, error=str(e))
            return False


def main():
    """Main entry point for cluster manager"""
    try:
        # Initialize logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        logger = structlog.get_logger(__name__)
        logger.info("Starting Cluster Manager")
        
        # Create and start cluster manager
        cluster_manager = ClusterManager()
        
        if not cluster_manager.start():
            logger.error("Failed to start cluster manager")
            return 1
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            cluster_manager.stop()
        
        logger.info("Cluster Manager shutdown complete")
        return 0
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
