"""
Worker Management System - High-level orchestration
Integrates worker algorithms with the model registry for smart model assignment
"""

from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
import structlog

from .schemas import ModelEntry, WorkerInfo, ModelType, ModelStatus, WorkerStatus
from .registry import ModelRegistry
from .worker_algorithms import WorkerLoadBalancer

logger = structlog.get_logger(__name__)

class WorkerManager:
    """High-level worker management with database integration"""
    
    def __init__(self, registry: ModelRegistry):
        """Initialize with model registry"""
        self.registry = registry
        self.load_balancer = WorkerLoadBalancer()
        
        logger.info("WorkerManager initialized")
    
    def assign_model_to_worker(self, model_id: str, 
                             preferred_worker_id: Optional[str] = None,
                             force: bool = False) -> Tuple[bool, str, Optional[str]]:
        """
        Assign a model to the best available worker
        
        Returns:
            (success, message, assigned_worker_id)
        """
        try:
            # Get model from registry
            model = self.registry.get_model(model_id)
            if not model:
                return False, f"Model {model_id} not found", None
            
            # Check if model is already assigned and loaded
            if model.status == ModelStatus.LOADED and model.assigned_worker and not force:
                return True, f"Model already loaded on worker {model.assigned_worker}", model.assigned_worker
            
            # Get available workers
            workers, _ = self.registry.list_workers(status=WorkerStatus.ONLINE)
            if not workers:
                return False, "No online workers available", None
            
            # If preferred worker specified, try it first
            if preferred_worker_id:
                preferred_worker = next((w for w in workers if w.id == preferred_worker_id), None)
                if preferred_worker:
                    if self._can_assign_model(model, preferred_worker):
                        return self._perform_assignment(model, preferred_worker)
                    else:
                        if force:
                            logger.warning("Forcing assignment to preferred worker", 
                                         worker_id=preferred_worker_id, model_id=model_id)
                            return self._perform_assignment(model, preferred_worker)
                        else:
                            logger.warning("Preferred worker cannot handle model", 
                                         worker_id=preferred_worker_id, model_id=model_id)
            
            # Use load balancer to find best worker
            selected_worker = self.load_balancer.select_optimal_worker(workers, model)
            if not selected_worker:
                return False, "No suitable worker found for model", None
            
            return self._perform_assignment(model, selected_worker)
            
        except Exception as e:
            logger.error("Failed to assign model to worker", model_id=model_id, error=str(e))
            return False, f"Assignment failed: {str(e)}", None
    
    def _can_assign_model(self, model: ModelEntry, worker: WorkerInfo) -> bool:
        """Check if worker can handle the model"""
        # Check memory requirements
        if model.size_gb > worker.memory_available_gb:
            return False
        
        # Check if worker has capacity for more models
        if len(worker.loaded_models) >= worker.max_models:
            return False
        
        # Check if model is already loaded on this worker
        if model.id in worker.loaded_models:
            return True  # Already loaded
        
        return True
    
    def _perform_assignment(self, model: ModelEntry, worker: WorkerInfo) -> Tuple[bool, str, str]:
        """Perform the actual model assignment"""
        try:
            # Update model assignment
            model.assigned_worker = worker.id
            model.status = ModelStatus.LOADING
            model.updated_at = datetime.now()
            
            # Update worker's loaded models if not already present
            if model.id not in worker.loaded_models:
                worker.loaded_models.append(model.id)
                worker.memory_used_gb += model.size_gb
                worker.memory_available_gb -= model.size_gb
            
            # Update database
            self.registry.update_model(model)
            self.registry.update_worker(worker)
            
            logger.info("Model assigned to worker", 
                       model_id=model.id, worker_id=worker.id, 
                       worker_memory_used=worker.memory_used_gb)
            
            return True, f"Model {model.id} assigned to worker {worker.id}", worker.id
            
        except Exception as e:
            logger.error("Failed to perform assignment", 
                        model_id=model.id, worker_id=worker.id, error=str(e))
            return False, f"Assignment update failed: {str(e)}", worker.id
    
    def unload_model_from_worker(self, model_id: str, worker_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Unload a model from its assigned worker
        
        Args:
            model_id: Model to unload
            worker_id: Specific worker (if None, uses model's assigned worker)
        """
        try:
            # Get model
            model = self.registry.get_model(model_id)
            if not model:
                return False, f"Model {model_id} not found"
            
            # Determine worker
            target_worker_id = worker_id or model.assigned_worker
            if not target_worker_id:
                return False, f"No worker specified for model {model_id}"
            
            # Get worker
            worker = self.registry.get_worker(target_worker_id)
            if not worker:
                return False, f"Worker {target_worker_id} not found"
            
            # Remove model from worker
            if model_id in worker.loaded_models:
                worker.loaded_models.remove(model_id)
                worker.memory_used_gb -= model.size_gb
                worker.memory_available_gb += model.size_gb
            
            # Update model status
            model.assigned_worker = None
            model.status = ModelStatus.AVAILABLE
            model.updated_at = datetime.now()
            
            # Update database
            self.registry.update_model(model)
            self.registry.update_worker(worker)
            
            logger.info("Model unloaded from worker", 
                       model_id=model_id, worker_id=target_worker_id)
            
            return True, f"Model {model_id} unloaded from worker {target_worker_id}"
            
        except Exception as e:
            logger.error("Failed to unload model", model_id=model_id, error=str(e))
            return False, f"Unload failed: {str(e)}"
    
    def rebalance_cluster(self, strategy: str = "memory") -> Dict[str, Any]:
        """
        Rebalance models across workers for optimal performance
        
        Args:
            strategy: Rebalancing strategy ("memory", "load", "models")
        """
        try:
            # Get all workers and models
            workers, _ = self.registry.list_workers(status=WorkerStatus.ONLINE)
            models, _ = self.registry.list_models(status=ModelStatus.LOADED)
            
            if not workers or not models:
                return {"success": True, "message": "No workers or models to rebalance", "moves": []}
            
            # Use load balancer to calculate optimal distribution
            recommendations = self.load_balancer.calculate_optimal_distribution(workers, models)
            
            moves = []
            errors = []
            
            # Execute recommended moves
            for move in recommendations:
                model_id = move["model_id"]
                from_worker = move["from_worker"]
                to_worker = move["to_worker"]
                
                # Unload from current worker
                if from_worker:
                    success, msg = self.unload_model_from_worker(model_id, from_worker)
                    if not success:
                        errors.append(f"Failed to unload {model_id} from {from_worker}: {msg}")
                        continue
                
                # Load to new worker
                success, msg, assigned_worker = self.assign_model_to_worker(model_id, to_worker, force=True)
                if success:
                    moves.append({
                        "model_id": model_id,
                        "from_worker": from_worker,
                        "to_worker": assigned_worker,
                        "reason": move.get("reason", "optimization")
                    })
                    logger.info("Rebalance move completed", 
                               model_id=model_id, from_worker=from_worker, to_worker=assigned_worker)
                else:
                    errors.append(f"Failed to assign {model_id} to {to_worker}: {msg}")
            
            return {
                "success": len(errors) == 0,
                "message": f"Rebalanced {len(moves)} models" + (f" with {len(errors)} errors" if errors else ""),
                "moves": moves,
                "errors": errors
            }
            
        except Exception as e:
            logger.error("Cluster rebalancing failed", error=str(e))
            return {"success": False, "message": f"Rebalancing failed: {str(e)}", "moves": []}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        try:
            # Get statistics from registry
            stats = self.registry.get_system_statistics()
            
            # Get workers with details
            workers, _ = self.registry.list_workers()
            worker_details = []
            
            for worker in workers:
                # Calculate load metrics
                memory_usage_pct = (worker.memory_used_gb / worker.memory_total_gb * 100) if worker.memory_total_gb > 0 else 0
                model_load_pct = (len(worker.loaded_models) / worker.max_models * 100) if worker.max_models > 0 else 0
                
                worker_details.append({
                    "id": worker.id,
                    "gpu_index": worker.gpu_index,
                    "hostname": worker.hostname,
                    "status": worker.status.value,
                    "memory_total_gb": worker.memory_total_gb,
                    "memory_used_gb": worker.memory_used_gb,
                    "memory_available_gb": worker.memory_available_gb,
                    "memory_usage_pct": round(memory_usage_pct, 1),
                    "loaded_models": worker.loaded_models,
                    "model_count": len(worker.loaded_models),
                    "max_models": worker.max_models,
                    "model_load_pct": round(model_load_pct, 1),
                    "last_heartbeat": worker.last_heartbeat.isoformat(),
                    "error_message": worker.error_message
                })
            
            # Calculate cluster-wide metrics
            if workers:
                total_memory = sum(w.memory_total_gb for w in workers)
                used_memory = sum(w.memory_used_gb for w in workers)
                cluster_memory_usage = (used_memory / total_memory * 100) if total_memory > 0 else 0
                
                total_model_capacity = sum(w.max_models for w in workers)
                total_loaded_models = sum(len(w.loaded_models) for w in workers)
                cluster_model_usage = (total_loaded_models / total_model_capacity * 100) if total_model_capacity > 0 else 0
            else:
                cluster_memory_usage = 0
                cluster_model_usage = 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cluster_metrics": {
                    "memory_usage_pct": round(cluster_memory_usage, 1),
                    "model_usage_pct": round(cluster_model_usage, 1),
                    "online_workers": len([w for w in workers if w.status == WorkerStatus.ONLINE]),
                    "total_workers": len(workers)
                },
                "registry_stats": stats,
                "workers": worker_details
            }
            
        except Exception as e:
            logger.error("Failed to get cluster status", error=str(e))
            return {"error": f"Failed to get status: {str(e)}"}
    
    def health_check_workers(self, heartbeat_timeout_minutes: int = 5) -> Dict[str, Any]:
        """
        Check worker health and mark stale workers as offline
        
        Args:
            heartbeat_timeout_minutes: Minutes after which a worker is considered stale
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=heartbeat_timeout_minutes)
            workers, _ = self.registry.list_workers()
            
            stale_workers = []
            healthy_workers = []
            
            for worker in workers:
                if worker.last_heartbeat < cutoff_time and worker.status != WorkerStatus.OFFLINE:
                    # Mark as offline
                    worker.status = WorkerStatus.OFFLINE
                    worker.error_message = f"No heartbeat since {worker.last_heartbeat.isoformat()}"
                    self.registry.update_worker(worker)
                    stale_workers.append(worker.id)
                    
                    logger.warning("Worker marked as offline due to stale heartbeat", 
                                 worker_id=worker.id, last_heartbeat=worker.last_heartbeat.isoformat())
                else:
                    healthy_workers.append(worker.id)
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "healthy_workers": healthy_workers,
                "stale_workers": stale_workers,
                "total_checked": len(workers)
            }
            
        except Exception as e:
            logger.error("Worker health check failed", error=str(e))
            return {"success": False, "error": f"Health check failed: {str(e)}"}
