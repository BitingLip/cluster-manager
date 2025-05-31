from celery import Celery
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun
import torch
import structlog
import time
import base64
import io
from PIL import Image
from typing import Dict, Any, Optional
import traceback
import psutil
import os
from pathlib import Path

from .config import settings
from .model_loader import model_manager

# Configure structured logging
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
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Generate worker ID if not provided
if settings.worker_id is None:
    settings.worker_id = f"worker-gpu{settings.gpu_index}-{os.getpid()}"

# Initialize Celery app
celery_app = Celery(
    'gpu_cluster_worker',  # Use a distinct name for worker
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery_task_serializer,
    result_serializer=settings.celery_result_serializer,
    accept_content=settings.celery_accept_content,
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_enable_utc,
    task_soft_time_limit=settings.task_soft_time_limit,
    task_time_limit=settings.task_timeout,
    worker_concurrency=settings.max_concurrent_tasks,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    # Explicit task discovery
    include=['app.tasks'],
    imports=['app.tasks'],
    # Use solo pool on Windows to avoid thread-local storage issues
    worker_pool='solo'
)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when worker is ready to accept tasks"""
    logger.info(
        "GPU worker ready",
        worker_id=settings.worker_id,
        gpu_index=settings.gpu_index,
        device=str(model_manager.device),
        pid=os.getpid()
    )


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Called when worker is shutting down"""
    logger.info("GPU worker shutting down", worker_id=settings.worker_id)


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Called before task execution"""
    logger.info(
        "Task started",
        task_id=task_id,
        task_name=task.name if task else "unknown",
        worker_id=settings.worker_id
    )


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Called after task execution"""
    logger.info(
        "Task completed",
        task_id=task_id,
        task_name=task.name if task else "unknown",
        state=state,
        worker_id=settings.worker_id
    )


def get_system_info() -> Dict[str, Any]:
    """Get current system information"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "worker_id": settings.worker_id,
            "gpu_index": settings.gpu_index,
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_percent": memory.percent,
            "pid": os.getpid(),
            "device": str(model_manager.device)
        }
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return {"worker_id": settings.worker_id, "error": str(e)}


@celery_app.task(name='app.tasks.run_llm_inference', bind=True)
def run_llm_inference(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run LLM inference task
    
    Args:
        task_data: Task data containing model_name and payload
        
    Returns:
        Dict containing the generated text and metadata
    """
    start_time = time.time()
    task_id = task_data.get("task_id", self.request.id)
    model_name = task_data.get("model_name", "gpt2")
    payload = task_data.get("payload", {})
    
    try:
        logger.info(
            "Starting LLM inference",
            task_id=task_id,
            model_name=model_name,
            worker_id=settings.worker_id
        )
        
        # Load model
        model_info = model_manager.load_llm_model(model_name)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        device = model_info["device"]
        
        # Extract parameters
        input_text = payload.get("text", "")
        max_tokens = min(payload.get("max_tokens", 50), settings.llm_max_length)
        temperature = payload.get("temperature", 0.7)
        top_p = payload.get("top_p", 0.9)
        repetition_penalty = payload.get("repetition_penalty", 1.1)
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.llm_max_length
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode output
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        execution_time = time.time() - start_time
        
        result = {
            "result": {
                "generated_text": generated_text,
                "input_text": input_text,
                "full_response": tokenizer.decode(outputs[0], skip_special_tokens=True)
            },
            "metadata": {
                "model_name": model_name,
                "execution_time": execution_time,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "input_length": len(input_text),
                "output_length": len(generated_text),
                "system_info": get_system_info()
            }
        }
        
        logger.info(
            "LLM inference completed",
            task_id=task_id,
            model_name=model_name,
            execution_time=execution_time,
            input_length=len(input_text),
            output_length=len(generated_text)
        )
        
        return result
        
    except Exception as e:
        error_msg = f"LLM inference failed: {str(e)}"
        logger.error(
            error_msg,
            task_id=task_id,
            model_name=model_name,
            error=str(e),
            traceback=traceback.format_exc()
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="app.tasks.run_sd_inference")
def run_sd_inference(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Stable Diffusion inference task
    
    Args:
        task_data: Task data containing model_name and payload
        
    Returns:
        Dict containing the generated image and metadata
    """
    start_time = time.time()
    task_id = task_data.get("task_id", self.request.id)
    model_name = task_data.get("model_name", "runwayml/stable-diffusion-v1-5")
    payload = task_data.get("payload", {})
    
    try:
        logger.info(
            "Starting Stable Diffusion inference",
            task_id=task_id,
            model_name=model_name,
            worker_id=settings.worker_id
        )
        
        # Load model
        model_info = model_manager.load_stable_diffusion_model(model_name)
        pipeline = model_info["pipeline"]
        
        # Extract parameters
        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", None)
        width = payload.get("width", settings.sd_image_size)
        height = payload.get("height", settings.sd_image_size)
        num_inference_steps = payload.get("num_inference_steps", settings.sd_num_inference_steps)
        guidance_scale = payload.get("guidance_scale", settings.sd_guidance_scale)
        seed = payload.get("seed", None)
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            result_img = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # Convert image to base64
        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        execution_time = time.time() - start_time
        
        result = {
            "result": {
                "image": img_base64,
                "format": "png",
                "width": width,
                "height": height
            },
            "metadata": {
                "model_name": model_name,
                "execution_time": execution_time,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "system_info": get_system_info()
            }
        }
        
        logger.info(
            "Stable Diffusion inference completed",
            task_id=task_id,
            model_name=model_name,
            execution_time=execution_time,
            image_size=f"{width}x{height}"
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Stable Diffusion inference failed: {str(e)}"
        logger.error(
            error_msg,
            task_id=task_id,
            model_name=model_name,
            error=str(e),
            traceback=traceback.format_exc()
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="app.tasks.run_tts_inference")
def run_tts_inference(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run TTS inference task
    
    Args:
        task_data: Task data containing model_name and payload
        
    Returns:
        Dict containing the generated audio and metadata
    """
    start_time = time.time()
    task_id = task_data.get("task_id", self.request.id)
    model_name = task_data.get("model_name", "microsoft/speecht5_tts")
    payload = task_data.get("payload", {})
    
    try:
        logger.info(
            "Starting TTS inference",
            task_id=task_id,
            model_name=model_name,
            worker_id=settings.worker_id
        )
        
        # Load model
        model_info = model_manager.load_tts_model(model_name)
        tts_pipeline = model_info["pipeline"]
        
        # Extract parameters
        text = payload.get("text", "")
        voice_id = payload.get("voice_id", None)
        speed = payload.get("speed", 1.0)
        pitch = payload.get("pitch", 1.0)
        
        # Generate speech
        audio_output = tts_pipeline(text)
        
        # Convert audio to base64 (implementation depends on the pipeline output format)
        # This is a simplified version - actual implementation would need to handle
        # the specific audio format returned by the TTS pipeline
        audio_base64 = ""  # Placeholder
        
        execution_time = time.time() - start_time
        
        result = {
            "result": {
                "audio": audio_base64,
                "format": settings.tts_audio_format,
                "sample_rate": settings.tts_sample_rate,
                "text": text
            },
            "metadata": {
                "model_name": model_name,
                "execution_time": execution_time,
                "text_length": len(text),
                "voice_id": voice_id,
                "speed": speed,
                "pitch": pitch,
                "system_info": get_system_info()
            }
        }
        
        logger.info(
            "TTS inference completed",
            task_id=task_id,
            model_name=model_name,
            execution_time=execution_time,
            text_length=len(text)
        )
        
        return result
        
    except Exception as e:
        error_msg = f"TTS inference failed: {str(e)}"
        logger.error(
            error_msg,
            task_id=task_id,
            model_name=model_name,
            error=str(e),
            traceback=traceback.format_exc()
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="app.tasks.run_image_to_text_inference")
def run_image_to_text_inference(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run image-to-text inference task
    
    Args:
        task_data: Task data containing model_name and payload
        
    Returns:
        Dict containing the generated caption and metadata
    """
    start_time = time.time()
    task_id = task_data.get("task_id", self.request.id)
    model_name = task_data.get("model_name", "Salesforce/blip-image-captioning-base")
    payload = task_data.get("payload", {})
    
    try:
        logger.info(
            "Starting image-to-text inference",
            task_id=task_id,
            model_name=model_name,
            worker_id=settings.worker_id
        )
        
        # Load model
        model_info = model_manager.load_image_to_text_model(model_name)
        img2txt_pipeline = model_info["pipeline"]
        
        # Extract parameters
        image_data = payload.get("image_data", "")
        max_tokens = payload.get("max_tokens", 100)
        
        # Decode image from base64
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
        
        # Generate caption
        caption_result = img2txt_pipeline(image, max_new_tokens=max_tokens)
        
        # Extract text from result
        if isinstance(caption_result, list) and len(caption_result) > 0:
            caption = caption_result[0].get("generated_text", "")
        else:
            caption = str(caption_result)
        
        execution_time = time.time() - start_time
        
        result = {
            "result": {
                "caption": caption,
                "image_size": image.size,
                "image_mode": image.mode
            },
            "metadata": {
                "model_name": model_name,
                "execution_time": execution_time,
                "max_tokens": max_tokens,
                "caption_length": len(caption),
                "system_info": get_system_info()
            }
        }
        
        logger.info(
            "Image-to-text inference completed",
            task_id=task_id,
            model_name=model_name,
            execution_time=execution_time,
            caption_length=len(caption)
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Image-to-text inference failed: {str(e)}"
        logger.error(
            error_msg,
            task_id=task_id,
            model_name=model_name,
            error=str(e),
            traceback=traceback.format_exc()
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="app.tasks.health_check")
def health_check(self) -> Dict[str, Any]:
    """
    Health check task to verify worker status
    
    Returns:
        Dict containing worker health information
    """
    try:
        system_info = get_system_info()
        cache_info = model_manager.get_cache_info()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system_info": system_info,
            "cache_info": cache_info,
            "worker_id": settings.worker_id
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "worker_id": settings.worker_id
        }


if __name__ == "__main__":
    # Start worker
    celery_app.start()
