"""
Cluster Manager Configuration
Uses the new distributed configuration system for microservice independence.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path to access config package
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from the config package (avoid circular imports)
from config.distributed_config import load_service_config, load_infrastructure_config
from config.service_discovery import ServiceDiscovery

class ClusterManagerSettings:
    """Cluster Manager specific configuration adapter using distributed config"""
    
    def __init__(self):
        # Load service-specific configuration
        self.config = load_service_config('cluster-manager', 'manager')
        
        # Load infrastructure configuration for shared resources
        self.infrastructure = load_infrastructure_config()
        
        # Initialize service discovery
        try:
            self.service_discovery = ServiceDiscovery()
        except Exception as e:
            print(f"Warning: Could not initialize service discovery: {e}")
            self.service_discovery = None
    
    def get_config_value(self, key: str, default: str = '') -> str:
        """Get configuration value with fallback to environment variables"""
        return self.config.get(key, os.getenv(key, default))
    
    @property
    def host(self):
        return self.get_config_value('CLUSTER_MANAGER_HOST', 'localhost')
    
    @property 
    def port(self):
        return int(self.get_config_value('CLUSTER_MANAGER_PORT', '8002'))
    
    @property
    def debug(self):
        return self.get_config_value('DEBUG', 'true').lower() == 'true'
    
    @property
    def cors_origins(self):
        origins = self.get_config_value('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173')
        return [origin.strip() for origin in origins.split(',')]
    
    @property
    def max_workers(self):
        return int(self.get_config_value('CLUSTER_MAX_WORKERS', '10'))
    
    @property
    def node_timeout(self):
        return int(self.get_config_value('CLUSTER_NODE_TIMEOUT', '30'))
    
    @property
    def db_host(self):
        return self.get_config_value('CLUSTER_DB_HOST', 'localhost')
    
    @property
    def db_port(self):
        return int(self.get_config_value('CLUSTER_DB_PORT', '5432'))
    
    @property
    def db_name(self):
        return self.get_config_value('CLUSTER_DB_NAME', 'bitinglip_cluster')
    
    @property
    def db_user(self):
        return self.get_config_value('CLUSTER_DB_USER', 'bitinglip')
    
    @property
    def db_password(self):
        return self.get_config_value('CLUSTER_DB_PASSWORD', 'secure_password')
    
    @property
    def log_level(self):
        return self.get_config_value('LOG_LEVEL', 'INFO')

# Create default instance
settings = ClusterManagerSettings()

# Backward compatibility alias
Settings = ClusterManagerSettings

# Re-export for backward compatibility
__all__ = ['settings', 'Settings', 'ClusterManagerSettings']
