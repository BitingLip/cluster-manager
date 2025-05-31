"""
Cluster Manager Configuration
Now uses centralized BitingLip configuration system.
"""

# Import from centralized configuration system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../config'))

from central_config import get_config
from service_discovery import ServiceDiscovery

class ClusterManagerSettings:
    """Cluster Manager specific configuration adapter"""
    
    def __init__(self):
        self.config = get_config('cluster_manager')
        self.service_discovery = ServiceDiscovery()
    
    @property
    def host(self):
        return self.config.cluster_manager_host
    
    @property 
    def port(self):
        return self.config.cluster_manager_port
    
    @property
    def debug(self):
        return self.config.debug
    
    @property
    def cors_origins(self):
        return self.config.cors_origins

# Create default instance
settings = ClusterManagerSettings()

# Backward compatibility alias
Settings = ClusterManagerSettings

# Re-export for backward compatibility
__all__ = ['settings', 'Settings', 'ClusterManagerSettings']
