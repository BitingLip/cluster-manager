import torch
import torch_directml as dml
from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, Any, Optional, Union
import os
from pathlib import Path
import time
import gc
import psutil
import structlog
from threading import Lock
import hashlib

from .config import settings

logger = structlog.get_logger(__name__)

# Global model cache with thread safety
_model_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = Lock()


class ModelManager:
    """Manages model loading, caching, and memory optimization"""
    
    def __init__(self):
        self.device = self._get_directml_device()
        self.cache_dir = Path(settings.model_cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(
            "ModelManager initialized",
            device=str(self.device),
            gpu_index=settings.gpu_index,
            cache_dir=str(self.cache_dir)        )
    
    def _get_directml_device(self) -> torch.device:
        """Get DirectML device for the configured GPU"""
        try:
            device = torch.device(dml.device(settings.gpu_index))
            logger.info(f"DirectML device initialized: {device}")
            return device
        except Exception as e:
            logger.error(f"Failed to initialize DirectML device: {e}")
            # Fallback to CPU
            return torch.device("cpu")
    
    def _get_model_key(self, model_name: str, model_type: str) -> str:
        """Generate a unique key for model caching"""
        return f"{model_type}:{model_name}"
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get the configured torch dtype"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(settings.torch_dtype, torch.float16)
    
    def _check_memory_usage(self) -> Dict[str, float]:
        """Check system and GPU memory usage"""
        try:
            # System memory
            sys_memory = psutil.virtual_memory()
            sys_usage = sys_memory.percent / 100.0
            
            # GPU memory (DirectML doesn't expose this easily, estimate from torch)
            gpu_usage = 0.0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            return {
                "system_memory_usage": sys_usage,
                "gpu_memory_usage": gpu_usage
            }
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return {"system_memory_usage": 0.0, "gpu_memory_usage": 0.0}
    
    def _cleanup_memory(self):
        """Force garbage collection and clear caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_llm_model(self, model_name: str) -> Dict[str, Any]:
        """Load a language model for text generation"""
        model_key = self._get_model_key(model_name, "llm")
        
        with _cache_lock:
            if model_key in _model_cache:
                logger.info(f"Using cached LLM model: {model_name}")
                return _model_cache[model_key]
        
        try:
            start_time = time.time()
            logger.info(f"Loading LLM model: {model_name}")
            
            # Check memory before loading
            memory_info = self._check_memory_usage()
            logger.info(f"Memory usage before loading: {memory_info}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self._get_torch_dtype(),
                trust_remote_code=True,
                device_map="auto" if str(self.device) != "cpu" else None
            )
            
            # Move to DirectML device
            if str(self.device) != "cpu":
                model = model.to(self.device)
            
            # Set to eval mode
            model.eval()
            
            load_time = time.time() - start_time
            
            model_info = {
                "model": model,
                "tokenizer": tokenizer,
                "device": self.device,
                "model_name": model_name,
                "load_time": load_time,
                "loaded_at": time.time()
            }
            
            with _cache_lock:
                _model_cache[model_key] = model_info
            
            logger.info(
                f"LLM model loaded successfully",
                model_name=model_name,
                load_time=load_time,
                device=str(self.device)
            )
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load LLM model {model_name}: {e}")
            raise
    
    def load_stable_diffusion_model(self, model_name: str) -> Dict[str, Any]:
        """Load a Stable Diffusion model for image generation"""
        model_key = self._get_model_key(model_name, "sd")
        
        with _cache_lock:
            if model_key in _model_cache:
                logger.info(f"Using cached SD model: {model_name}")
                return _model_cache[model_key]
        
        try:
            start_time = time.time()
            logger.info(f"Loading Stable Diffusion model: {model_name}")
            
            # Load pipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self._get_torch_dtype(),
                safety_checker=None,  # Disable for performance
                requires_safety_checker=False
            )
            
            # Move to DirectML device
            if str(self.device) != "cpu":
                pipe = pipe.to(self.device)
            
            # Enable memory efficient attention
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            
            load_time = time.time() - start_time
            
            model_info = {
                "pipeline": pipe,
                "device": self.device,
                "model_name": model_name,
                "load_time": load_time,
                "loaded_at": time.time()
            }
            
            with _cache_lock:
                _model_cache[model_key] = model_info
            
            logger.info(
                f"Stable Diffusion model loaded successfully",
                model_name=model_name,
                load_time=load_time,
                device=str(self.device)
            )
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model {model_name}: {e}")
            raise
    
    def load_tts_model(self, model_name: str) -> Dict[str, Any]:
        """Load a TTS model for speech synthesis"""
        model_key = self._get_model_key(model_name, "tts")
        
        with _cache_lock:
            if model_key in _model_cache:
                logger.info(f"Using cached TTS model: {model_name}")
                return _model_cache[model_key]
        
        try:
            start_time = time.time()
            logger.info(f"Loading TTS model: {model_name}")
            
            # Create TTS pipeline
            tts_pipeline = pipeline(
                "text-to-speech",
                model=model_name,
                device=self.device if str(self.device) != "cpu" else -1,
                torch_dtype=self._get_torch_dtype(),
                model_kwargs={"cache_dir": self.cache_dir}
            )
            
            load_time = time.time() - start_time
            
            model_info = {
                "pipeline": tts_pipeline,
                "device": self.device,
                "model_name": model_name,
                "load_time": load_time,
                "loaded_at": time.time()
            }
            
            with _cache_lock:
                _model_cache[model_key] = model_info
            
            logger.info(
                f"TTS model loaded successfully",
                model_name=model_name,
                load_time=load_time,
                device=str(self.device)
            )
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load TTS model {model_name}: {e}")
            raise
    
    def load_image_to_text_model(self, model_name: str) -> Dict[str, Any]:
        """Load an image-to-text model for captioning"""
        model_key = self._get_model_key(model_name, "img2txt")
        
        with _cache_lock:
            if model_key in _model_cache:
                logger.info(f"Using cached image-to-text model: {model_name}")
                return _model_cache[model_key]
        
        try:
            start_time = time.time()
            logger.info(f"Loading image-to-text model: {model_name}")
            
            # Create image-to-text pipeline
            img2txt_pipeline = pipeline(
                "image-to-text",
                model=model_name,
                device=self.device if str(self.device) != "cpu" else -1,
                torch_dtype=self._get_torch_dtype(),
                model_kwargs={"cache_dir": self.cache_dir}
            )
            
            load_time = time.time() - start_time
            
            model_info = {
                "pipeline": img2txt_pipeline,
                "device": self.device,
                "model_name": model_name,
                "load_time": load_time,
                "loaded_at": time.time()
            }
            
            with _cache_lock:
                _model_cache[model_key] = model_info
            
            logger.info(
                f"Image-to-text model loaded successfully",
                model_name=model_name,
                load_time=load_time,
                device=str(self.device)
            )
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load image-to-text model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str, model_type: str):
        """Unload a model from cache to free memory"""
        model_key = self._get_model_key(model_name, model_type)
        
        with _cache_lock:
            if model_key in _model_cache:
                del _model_cache[model_key]
                logger.info(f"Unloaded model: {model_key}")
        
        self._cleanup_memory()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        with _cache_lock:
            return {
                "cached_models": len(_model_cache),
                "models": [
                    {
                        "key": key,
                        "model_name": info.get("model_name", "unknown"),
                        "loaded_at": info.get("loaded_at", 0),
                        "load_time": info.get("load_time", 0)
                    }
                    for key, info in _model_cache.items()
                ]
            }


# Global model manager instance
model_manager = ModelManager()
