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
        """Update GPU resource information for both NVIDIA and AMD GPUs"""
        try:
            from .gpu_detector import GPUDetector
            
            detector = GPUDetector()
            gpu_data = detector.detect_all_gpus()
            
            # Update database if we have GPU data
            if gpu_data:
                # Store each GPU's information in the database
                for gpu in gpu_data:
                    try:
                        self.database.update_gpu_resources(
                            node_id=self.node_id,
                            gpu_data=gpu
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store GPU {gpu.get('index', 'unknown')} data: {e}")
                
                logger.debug(f"Updated GPU resources for {len(gpu_data)} GPU(s)")
            else:
                logger.debug("No GPUs detected or available")
                
        except Exception as e:
            logger.warning(f"Failed to collect GPU information: {e}")
    
    def get_gpu_summary(self) -> Dict[str, Any]:
        """Get a summary of GPU capabilities for AI workloads"""
        try:
            from .gpu_detector import GPUDetector
            
            detector = GPUDetector()
            gpu_data = detector.detect_all_gpus()
            
            summary = {
                'total_gpus': len(gpu_data),
                'nvidia_gpus': 0,
                'amd_gpus': 0,
                'total_vram_gb': 0,
                'ai_capable_gpus': 0,
                'training_capable_gpus': 0,
                'inference_capable_gpus': 0,
                'recommended_frameworks': set(),
                'supported_models': set(),
                'performance_tiers': {},
                'gpu_details': []
            }
            
            for gpu in gpu_data:
                # Count by vendor
                if gpu['vendor'] == 'NVIDIA':
                    summary['nvidia_gpus'] += 1
                elif gpu['vendor'] == 'AMD':
                    summary['amd_gpus'] += 1
                
                # Sum VRAM
                summary['total_vram_gb'] += gpu.get('memory_total_mb', 0) / 1024
                
                # Analyze AI capabilities
                ai_caps = gpu.get('ai_capabilities', {})
                
                if ai_caps.get('suitable_for_inference'):
                    summary['inference_capable_gpus'] += 1
                
                if ai_caps.get('suitable_for_training'):
                    summary['training_capable_gpus'] += 1
                
                if ai_caps.get('suitable_for_inference') or ai_caps.get('suitable_for_training'):
                    summary['ai_capable_gpus'] += 1
                
                # Collect frameworks and models
                frameworks = gpu.get('framework_support', [])
                summary['recommended_frameworks'].update(frameworks)
                
                models = ai_caps.get('recommended_models', [])
                summary['supported_models'].update(models)
                
                # Performance tiers
                tier = ai_caps.get('performance_tier', 'unknown')
                summary['performance_tiers'][tier] = summary['performance_tiers'].get(tier, 0) + 1
                
                # GPU details for summary
                summary['gpu_details'].append({
                    'name': gpu['name'],
                    'vendor': gpu['vendor'],
                    'memory_gb': gpu.get('memory_total_mb', 0) / 1024,
                    'architecture': gpu.get('metadata', {}).get('architecture', 'Unknown'),
                    'ai_performance_tier': tier,
                    'suitable_for_training': ai_caps.get('suitable_for_training', False),
                    'suitable_for_inference': ai_caps.get('suitable_for_inference', False)
                })
            
            # Convert sets to lists for JSON serialization
            summary['recommended_frameworks'] = list(summary['recommended_frameworks'])
            summary['supported_models'] = list(summary['supported_models'])
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate GPU summary: {e}")
            return {}
    
    def _detect_nvidia_gpus(self) -> List[Dict[str, Any]]:
        """Detect NVIDIA GPUs using NVML"""
        nvidia_gpus = []
        
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
                
                # Get compute capability and other details
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "Unknown"
                
                # Determine AI capabilities based on architecture
                ai_capabilities = self._assess_nvidia_ai_capabilities(name, compute_capability, memory_total_mb)
                
                gpu_info = {
                    'index': i,
                    'vendor': 'NVIDIA',
                    'name': name,
                    'memory_total_mb': memory_total_mb,
                    'memory_used_mb': memory_used_mb,
                    'memory_available_mb': memory_available_mb,
                    'utilization_percent': utilization_percent,
                    'temperature_celsius': temperature,
                    'power_usage_watts': power_usage,
                    'compute_capability': compute_capability,
                    'ai_capabilities': ai_capabilities,
                    'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                    'framework_support': ['CUDA', 'DirectML', 'OpenCL'],
                    'metadata': {
                        'collection_time': datetime.now().isoformat(),
                        'detection_method': 'pynvml',
                        'architecture': self._get_nvidia_architecture(name)
                    }
                }
                
                nvidia_gpus.append(gpu_info)
            
            if nvidia_gpus:
                logger.info(f"Detected {len(nvidia_gpus)} NVIDIA GPU(s)")
                
        except ImportError:
            logger.debug("pynvml not available, skipping NVIDIA GPU detection")
        except Exception as e:
            logger.warning("Failed to detect NVIDIA GPUs", error=str(e))
        
        return nvidia_gpus
    
    def _detect_amd_gpus(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using Windows DirectML and WMI"""
        amd_gpus = []
        
        # Method 1: Try WMI for GPU enumeration (Windows)
        if sys.platform == 'win32':
            amd_gpus.extend(self._detect_amd_gpus_wmi())
        
        # Method 2: Try DirectML device enumeration
        amd_gpus.extend(self._detect_amd_gpus_directml())
        
        # Method 3: Try OpenCL (cross-platform)
        amd_gpus.extend(self._detect_amd_gpus_opencl())
        
        # Remove duplicates based on name and keep the most detailed info
        unique_gpus = {}
        for gpu in amd_gpus:
            key = gpu['name']
            if key not in unique_gpus or len(gpu) > len(unique_gpus[key]):
                unique_gpus[key] = gpu
        
        final_amd_gpus = list(unique_gpus.values())
        if final_amd_gpus:
            logger.info(f"Detected {len(final_amd_gpus)} AMD GPU(s)")
        
        return final_amd_gpus
    
    def _detect_amd_gpus_wmi(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using Windows WMI"""
        amd_gpus = []
        
        try:
            import wmi
            c = wmi.WMI()
            
            for gpu in c.Win32_VideoController():
                if gpu.Name and 'AMD' in gpu.Name or 'Radeon' in gpu.Name:
                    # Get available memory (if possible)
                    memory_mb = 0
                    if gpu.AdapterRAM:
                        memory_mb = gpu.AdapterRAM // (1024 * 1024)
                    
                    # Assess AI capabilities based on GPU model
                    ai_capabilities = self._assess_amd_ai_capabilities(gpu.Name, memory_mb)
                    
                    gpu_info = {
                        'index': len(amd_gpus),
                        'vendor': 'AMD',
                        'name': gpu.Name,
                        'memory_total_mb': memory_mb,
                        'memory_used_mb': 0,  # WMI doesn't provide current usage
                        'memory_available_mb': memory_mb,
                        'utilization_percent': 0,  # WMI doesn't provide utilization
                        'temperature_celsius': None,
                        'power_usage_watts': None,
                        'driver_version': gpu.DriverVersion or 'Unknown',
                        'ai_capabilities': ai_capabilities,
                        'framework_support': self._get_amd_framework_support(gpu.Name),
                        'metadata': {
                            'collection_time': datetime.now().isoformat(),
                            'detection_method': 'wmi',
                            'architecture': self._get_amd_architecture(gpu.Name),
                            'pnp_device_id': gpu.PNPDeviceID
                        }
                    }
                    amd_gpus.append(gpu_info)
            
        except ImportError:
            logger.debug("WMI not available for AMD GPU detection")
        except Exception as e:
            logger.warning("Failed to detect AMD GPUs via WMI", error=str(e))
        
        return amd_gpus
    
    def _detect_amd_gpus_directml(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using DirectML (Windows)"""
        amd_gpus = []
        
        try:
            # This would require DirectML Python bindings
            # For now, we'll use a placeholder approach
            logger.debug("DirectML GPU detection not implemented yet")
        except Exception as e:
            logger.debug("DirectML AMD GPU detection failed", error=str(e))
        
        return amd_gpus
    
    def _detect_amd_gpus_opencl(self) -> List[Dict[str, Any]]:
        """Detect AMD GPUs using OpenCL"""
        amd_gpus = []
        
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            for platform in platforms:
                if 'AMD' in platform.name or 'Advanced Micro Devices' in platform.name:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    
                    for i, device in enumerate(devices):
                        memory_mb = device.global_mem_size // (1024 * 1024)
                        
                        # Assess AI capabilities
                        ai_capabilities = self._assess_amd_ai_capabilities(device.name, memory_mb)
                        
                        gpu_info = {
                            'index': len(amd_gpus),
                            'vendor': 'AMD',
                            'name': device.name,
                            'memory_total_mb': memory_mb,
                            'memory_used_mb': 0,  # OpenCL doesn't provide usage
                            'memory_available_mb': memory_mb,
                            'utilization_percent': 0,
                            'temperature_celsius': None,
                            'power_usage_watts': None,
                            'compute_units': device.max_compute_units,
                            'max_work_group_size': device.max_work_group_size,
                            'ai_capabilities': ai_capabilities,
                            'framework_support': self._get_amd_framework_support(device.name),
                            'metadata': {
                                'collection_time': datetime.now().isoformat(),
                                'detection_method': 'opencl',
                                'architecture': self._get_amd_architecture(device.name),
                                'opencl_version': device.version,
                                'platform': platform.name
                            }
                        }
                        amd_gpus.append(gpu_info)
            
        except ImportError:
            logger.debug("PyOpenCL not available for AMD GPU detection")
        except Exception as e:
            logger.debug("OpenCL AMD GPU detection failed", error=str(e))
        
        return amd_gpus
    
    def _assess_nvidia_ai_capabilities(self, name: str, compute_capability: str, memory_mb: int) -> Dict[str, Any]:
        """Assess AI capabilities of NVIDIA GPU"""
        capabilities = {
            'tensor_cores': False,
            'fp16_support': False,
            'int8_support': False,
            'suitable_for_inference': False,
            'suitable_for_training': False,
            'recommended_models': []
        }
        
        # Parse compute capability
        try:
            major = int(compute_capability.split('.')[0])
            minor = int(compute_capability.split('.')[1])
        except:
            major, minor = 0, 0
        
        # Tensor cores available from compute capability 7.0+
        if major >= 7:
            capabilities['tensor_cores'] = True
            capabilities['fp16_support'] = True
            capabilities['int8_support'] = True
        elif major >= 6:
            capabilities['fp16_support'] = True
        
        # Memory-based recommendations
        if memory_mb >= 24000:  # 24GB+
            capabilities['suitable_for_training'] = True
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-65B-4bit', 'Stable Diffusion XL', 'GPT-3.5 equivalent']
        elif memory_mb >= 16000:  # 16GB+
            capabilities['suitable_for_training'] = True
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-30B-4bit', 'Stable Diffusion 2.1', 'Fine-tuning 7B models']
        elif memory_mb >= 8000:  # 8GB+
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-13B-4bit', 'Stable Diffusion 1.5', 'Small model training']
        elif memory_mb >= 4000:  # 4GB+
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-7B-4bit', 'Stable Diffusion (optimized)', 'Small CNNs']
        
        return capabilities
    
    def _assess_amd_ai_capabilities(self, name: str, memory_mb: int) -> Dict[str, Any]:
        """Assess AI capabilities of AMD GPU based on architecture and specs"""
        capabilities = {
            'directml_support': True,  # All DX12 AMD GPUs support DirectML
            'fp16_support': False,
            'quantization_support': False,
            'driver_optimized': False,
            'suitable_for_inference': False,
            'suitable_for_training': False,
            'recommended_models': [],
            'architecture_notes': ''
        }
        
        # Determine architecture and capabilities
        arch = self._get_amd_architecture(name)
        
        if 'RDNA3' in arch:
            capabilities.update({
                'fp16_support': True,
                'quantization_support': True,
                'driver_optimized': True,
                'ai_accelerators': True,
                'architecture_notes': 'Latest AMD architecture with AI accelerators and optimized DirectML drivers'
            })
        elif 'RDNA2' in arch:
            capabilities.update({
                'fp16_support': True,
                'quantization_support': True,
                'driver_optimized': True,
                'architecture_notes': 'Good AI performance with DirectML optimizations and quantization support'
            })
        elif 'RDNA1' in arch:
            capabilities.update({
                'fp16_support': True,
                'quantization_support': True,
                'driver_optimized': True,
                'architecture_notes': 'Moderate AI performance with DirectML optimizations'
            })
        elif 'Vega' in arch:
            capabilities.update({
                'fp16_support': True,  # Vega has "rapid packed math"
                'quantization_support': False,
                'driver_optimized': False,
                'architecture_notes': 'Legacy architecture with basic DirectML support, no recent AI optimizations'
            })
        elif 'Polaris' in arch:
            capabilities.update({
                'fp16_support': False,  # FP32 only
                'quantization_support': False,
                'driver_optimized': False,
                'architecture_notes': 'Legacy architecture with basic DirectML support, FP32 only'
            })
        
        # Memory-based recommendations (similar to NVIDIA but adjusted for AMD)
        if memory_mb >= 20000:  # 20GB+ (RX 7900 XT/XTX)
            capabilities['suitable_for_training'] = True
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-30B-4bit', 'Stable Diffusion XL', 'Large model fine-tuning']
        elif memory_mb >= 16000:  # 16GB (RX 6800 XT, Radeon VII)
            capabilities['suitable_for_training'] = True
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-13B-4bit', 'Stable Diffusion 2.1', 'Medium model training']
        elif memory_mb >= 12000:  # 12GB (RX 6700 XT)
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-13B-4bit', 'Stable Diffusion XL (optimized)', 'Small model training']
        elif memory_mb >= 8000:  # 8GB (RX 580, RX 5700, RX 6600)
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['LLaMA-7B-4bit', 'Stable Diffusion 1.5', 'Small model inference']
        elif memory_mb >= 4000:  # 4GB (RX 580 4GB, RX 470)
            capabilities['suitable_for_inference'] = True
            capabilities['recommended_models'] = ['Phi-3 mini', 'Stable Diffusion (low VRAM mode)', 'Basic CNNs']
        
        return capabilities
    
    def _get_nvidia_architecture(self, name: str) -> str:
        """Determine NVIDIA GPU architecture from name"""
        name_upper = name.upper()
        
        if 'RTX 40' in name_upper or 'RTX 4' in name_upper:
            return 'Ada Lovelace'
        elif 'RTX 30' in name_upper or 'RTX 3' in name_upper:
            return 'Ampere'
        elif 'RTX 20' in name_upper or 'RTX 2' in name_upper:
            return 'Turing'
        elif 'GTX 16' in name_upper:
            return 'Turing (GTX)'
        elif 'GTX 10' in name_upper:
            return 'Pascal'
        elif 'GTX 9' in name_upper:
            return 'Maxwell'
        elif 'GTX 7' in name_upper or 'GTX 6' in name_upper:
            return 'Kepler'
        else:
            return 'Unknown'
    
    def _get_amd_architecture(self, name: str) -> str:
        """Determine AMD GPU architecture from name"""
        name_upper = name.upper()
        
        if 'RX 7' in name_upper or '7900' in name_upper or '7800' in name_upper or '7700' in name_upper or '7600' in name_upper:
            return 'RDNA3'
        elif 'RX 6' in name_upper or '6900' in name_upper or '6800' in name_upper or '6700' in name_upper or '6600' in name_upper or '6500' in name_upper:
            return 'RDNA2'
        elif 'RX 5' in name_upper or '5700' in name_upper or '5600' in name_upper or '5500' in name_upper:
            return 'RDNA1'
        elif 'VEGA' in name_upper or 'RADEON VII' in name_upper:
            return 'Vega (GCN 5)'
        elif 'RX 5' in name_upper and ('80' in name_upper or '70' in name_upper):  # RX 580, 570
            return 'Polaris (GCN 4)'
        elif 'RX 4' in name_upper:  # RX 480, 470
            return 'Polaris (GCN 4)'
        elif 'R9' in name_upper or 'R7' in name_upper:
            return 'GCN (Legacy)'
        else:
            return 'Unknown'
    
    def _get_amd_framework_support(self, name: str) -> List[str]:
        """Get supported frameworks for AMD GPU"""
        frameworks = ['DirectML']  # All modern AMD GPUs support DirectML
        
        arch = self._get_amd_architecture(name)
        
        # Add framework support based on architecture
        if 'RDNA' in arch:
            frameworks.extend(['PyTorch-DirectML', 'ONNX-DirectML', 'TensorFlow-DirectML'])
        
        # OpenCL support is universal
        frameworks.append('OpenCL')
        
        # ROCm support (Linux, newer cards)
        if 'RDNA2' in arch or 'RDNA3' in arch:
            frameworks.append('ROCm (Linux)')
        
        return frameworks
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
