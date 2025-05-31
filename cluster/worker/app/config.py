from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Worker configuration"""
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Celery Configuration
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_accept_content: List[str] = ["json"]
    celery_timezone: str = "UTC"
    celery_enable_utc: bool = True
    
    # GPU Configuration
    gpu_index: int = 0
    directml_device_index: Optional[int] = None  # If None, uses gpu_index
    gpu_memory_fraction: float = 0.9  # Use 90% of GPU memory
      # Model Configuration
    model_cache_dir: str = "../../../model-manager/models"
    max_model_cache_size_gb: float = 50.0  # Maximum cache size in GB
    model_load_timeout: int = 300  # 5 minutes to load a model
    
    # Task Configuration
    max_concurrent_tasks: int = 1  # One task per GPU
    task_timeout: int = 300  # 5 minutes default timeout
    task_soft_time_limit: int = 270  # 4.5 minutes soft limit
    
    # Performance Configuration
    inference_batch_size: int = 1
    torch_dtype: str = "float16"  # float16, float32, bfloat16
    attention_implementation: str = "sdpa"  # sdpa, flash_attention_2
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    worker_id: Optional[str] = None  # Auto-generated if None
    
    # DirectML Configuration
    directml_debug: bool = False
    directml_enable_graph_capture: bool = True
    directml_enable_metacommands: bool = True
    
    # Model-specific settings
    llm_max_length: int = 2048
    llm_pad_token_id: Optional[int] = None
    
    tts_sample_rate: int = 22050
    tts_audio_format: str = "wav"
    
    sd_image_size: int = 512
    sd_num_inference_steps: int = 20
    sd_guidance_scale: float = 7.5
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    memory_warning_threshold: float = 0.85  # 85% memory usage warning
    temperature_warning_threshold: float = 80.0  # 80°C warning
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Set DirectML environment variables
if settings.directml_device_index is not None:
    os.environ["TORCH_DIRECTML_DEVICE"] = str(settings.directml_device_index)
else:
    os.environ["TORCH_DIRECTML_DEVICE"] = str(settings.gpu_index)

if settings.directml_debug:
    os.environ["DIRECTML_DEBUG"] = "1"

if settings.directml_enable_graph_capture:
    os.environ["DIRECTML_ENABLE_GRAPH_CAPTURE"] = "1"

if settings.directml_enable_metacommands:
    os.environ["DIRECTML_ENABLE_METACOMMANDS"] = "1"
