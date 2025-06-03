"""
Simple Cluster Manager Configuration
Minimal configuration loader for cluster manager without complex dependencies.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

class ClusterManagerSettings:
    """Simple cluster manager configuration"""
    
    def __init__(self):
        # Load environment variables from cluster-manager-db.env
        env_file = Path(__file__).parent.parent / "config" / "cluster-manager-db.env"
        if env_file.exists():
            load_dotenv(env_file)
            
        # Load any additional environment files
        env_file2 = Path(__file__).parent.parent / "config" / "cluster-manager.env"
        if env_file2.exists():
            load_dotenv(env_file2)
    
    def get_config_value(self, key: str, default: str = '') -> str:
        """Get configuration value from environment variables"""
        return os.getenv(key, default)
    
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
    def db_host(self):
        return self.get_config_value('CLUSTER_DB_HOST', 'localhost')
    
    @property
    def db_port(self):
        return self.get_config_value('CLUSTER_DB_PORT', '5432')
    
    @property
    def db_name(self):
        return self.get_config_value('CLUSTER_DB_NAME', 'bitinglip_cluster')
    
    @property
    def db_user(self):
        return self.get_config_value('CLUSTER_DB_USER', 'postgres')
    
    @property
    def db_password(self):
        return self.get_config_value('CLUSTER_DB_PASSWORD', 'password')


# Create default instance
settings = ClusterManagerSettings()

# For backwards compatibility
Settings = ClusterManagerSettings

__all__ = ['settings', 'Settings', 'ClusterManagerSettings']
