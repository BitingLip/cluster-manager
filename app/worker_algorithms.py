"""
Worker Assignment Algorithms
Smart algorithms for distributing models across AMD GPU workers
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import structlog

from .schemas import ModelEntry, WorkerInfo, ModelType, ModelStatus, WorkerStatus

logger = structlog.get_logger(__name__)

@dataclass
class AssignmentScore:
    """Score for worker-model assignment"""
    worker_id: str
    score: float
    memory_fit: float
    load_balance: float
    affinity: float
    capacity: float
    reasons: List[str]

class WorkerLoadBalancer:
    """Intelligent load balancing for model assignments"""
    
    def __init__(self):
        """Initialize load balancer"""
        self.weights = {
            "memory_fit": 0.4,      # How well memory requirements fit
            "load_balance": 0.3,    # Current load distribution
            "affinity": 0.2,        # Model type affinity
            "capacity": 0.1         # Future capacity considerations
        }
        
        logger.info("WorkerLoadBalancer initialized", weights=self.weights)
    
    def select_optimal_worker(self, workers: List[WorkerInfo], model: ModelEntry) -> Optional[WorkerInfo]:
        """
        Select the best worker for a model using weighted scoring
        
        Args:
            workers: Available workers
            model: Model to assign
            
        Returns:
            Best worker or None if no suitable worker
        """
        if not workers:
            return None
        
        # Filter workers that can handle the model
        suitable_workers = [w for w in workers if self._can_worker_handle_model(w, model)]
        if not suitable_workers:
            logger.warning("No suitable workers found", model_id=model.id, 
                         model_size_gb=model.size_gb, available_workers=len(workers))
            return None
        
        # Score all suitable workers
        scores = []
        for worker in suitable_workers:
            score = self._calculate_assignment_score(worker, model, suitable_workers)
            scores.append(score)
        
        # Select highest scoring worker
        best_score = max(scores, key=lambda s: s.score)
        best_worker = next(w for w in suitable_workers if w.id == best_score.worker_id)
        
        logger.info("Optimal worker selected", 
                   model_id=model.id, worker_id=best_worker.id, 
                   score=round(best_score.score, 3), reasons=best_score.reasons)
        
        return best_worker
    
    def _can_worker_handle_model(self, worker: WorkerInfo, model: ModelEntry) -> bool:
        """Check if worker can handle the model"""
        # Worker must be online
        if worker.status != WorkerStatus.ONLINE:
            return False
        
        # Must have enough memory
        if model.size_gb > worker.memory_available_gb:
            return False
        
        # Must have capacity for more models
        if len(worker.loaded_models) >= worker.max_models:
            return False
        
        return True
    
    def _calculate_assignment_score(self, worker: WorkerInfo, model: ModelEntry, 
                                  all_workers: List[WorkerInfo]) -> AssignmentScore:
        """Calculate comprehensive assignment score"""
        reasons = []
        
        # 1. Memory fit score (0-1)
        memory_utilization = worker.memory_used_gb / worker.memory_total_gb
        memory_after = (worker.memory_used_gb + model.size_gb) / worker.memory_total_gb
        
        # Prefer workers with good memory utilization but not overloaded
        if memory_after > 0.9:
            memory_fit = 0.1  # Too much memory usage
            reasons.append("high_memory_usage")
        elif memory_after > 0.7:
            memory_fit = 0.6  # Acceptable but high
            reasons.append("moderate_memory_usage")
        else:
            memory_fit = 1.0 - memory_utilization  # Better with more free memory
            reasons.append("good_memory_fit")
        
        # 2. Load balance score (0-1)
        current_load = len(worker.loaded_models) / worker.max_models if worker.max_models > 0 else 0
        avg_load = sum(len(w.loaded_models) / w.max_models for w in all_workers if w.max_models > 0) / len(all_workers)
        
        # Prefer workers with lower than average load
        if current_load < avg_load:
            load_balance = 1.0 - current_load
            reasons.append("below_avg_load")
        else:
            load_balance = max(0.1, 1.0 - (current_load - avg_load))
            reasons.append("above_avg_load")
        
        # 3. Model type affinity score (0-1)
        affinity = self._calculate_type_affinity(worker, model)
        if affinity > 0.7:
            reasons.append("good_type_affinity")
        elif affinity < 0.3:
            reasons.append("poor_type_affinity")
        
        # 4. Capacity score (0-1)
        remaining_capacity = (worker.max_models - len(worker.loaded_models) - 1) / worker.max_models
        capacity = max(0.1, remaining_capacity)
        if capacity > 0.5:
            reasons.append("good_capacity")
        
        # Calculate weighted total score
        total_score = (
            memory_fit * self.weights["memory_fit"] +
            load_balance * self.weights["load_balance"] +
            affinity * self.weights["affinity"] +
            capacity * self.weights["capacity"]
        )
        
        return AssignmentScore(
            worker_id=worker.id,
            score=total_score,
            memory_fit=memory_fit,
            load_balance=load_balance,
            affinity=affinity,
            capacity=capacity,
            reasons=reasons
        )
    
    def _calculate_type_affinity(self, worker: WorkerInfo, model: ModelEntry) -> float:
        """Calculate affinity between worker and model type"""
        # Check what types of models this worker already has
        loaded_types = set()
        
        # For now, assume all models are compatible
        # In a real system, you'd check worker capabilities
        
        # Slight preference for workers already running similar model types
        # This could be extended with GPU-specific optimization
        if model.type == ModelType.LLM:
            return 0.8  # LLMs work well on most GPUs
        elif model.type == ModelType.VISION:
            return 0.9  # Vision models often have good GPU acceleration
        elif model.type == ModelType.DIFFUSION:
            return 0.7  # Diffusion models can be memory intensive
        else:
            return 0.6  # Other types
    
    def calculate_optimal_distribution(self, workers: List[WorkerInfo], 
                                     models: List[ModelEntry]) -> List[Dict[str, Any]]:
        """
        Calculate optimal redistribution of models across workers
        
        Returns:
            List of recommended moves: [{"model_id": str, "from_worker": str, "to_worker": str, "reason": str}]
        """
        recommendations = []
        
        if not workers or not models:
            return recommendations
        
        # Calculate current load distribution
        worker_loads = {}
        for worker in workers:
            if worker.status == WorkerStatus.ONLINE:
                load = len(worker.loaded_models) / worker.max_models if worker.max_models > 0 else 0
                memory_usage = worker.memory_used_gb / worker.memory_total_gb if worker.memory_total_gb > 0 else 0
                worker_loads[worker.id] = {"load": load, "memory": memory_usage, "worker": worker}
        
        if not worker_loads:
            return recommendations
        
        # Find overloaded and underloaded workers
        avg_load = sum(data["load"] for data in worker_loads.values()) / len(worker_loads)
        avg_memory = sum(data["memory"] for data in worker_loads.values()) / len(worker_loads)
        
        overloaded_workers = [
            (wid, data) for wid, data in worker_loads.items() 
            if data["load"] > avg_load + 0.2 or data["memory"] > 0.8
        ]
        
        underloaded_workers = [
            (wid, data) for wid, data in worker_loads.items()
            if data["load"] < avg_load - 0.2 and data["memory"] < 0.6
        ]
        
        # Recommend moves from overloaded to underloaded workers
        for overloaded_id, overloaded_data in overloaded_workers:
            overloaded_worker = overloaded_data["worker"]
            
            # Find models that could be moved
            moveable_models = [
                m for m in models 
                if m.assigned_worker == overloaded_id and m.status == ModelStatus.LOADED
            ]
            
            for model in moveable_models:
                # Find best underloaded worker for this model
                best_target = None
                best_score = 0
                
                for underloaded_id, underloaded_data in underloaded_workers:
                    underloaded_worker = underloaded_data["worker"]
                    
                    if self._can_worker_handle_model(underloaded_worker, model):
                        # Calculate benefit of move
                        score = self._calculate_move_benefit(
                            model, overloaded_worker, underloaded_worker
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_target = underloaded_worker
                
                if best_target and best_score > 0.3:  # Only recommend if significant benefit
                    reason = "load_balancing"
                    if overloaded_data["memory"] > 0.8:
                        reason = "memory_pressure"
                    
                    recommendations.append({
                        "model_id": model.id,
                        "from_worker": overloaded_id,
                        "to_worker": best_target.id,
                        "reason": reason,
                        "benefit_score": round(best_score, 3)
                    })
                    
                    # Update tracking to avoid multiple moves to same worker
                    underloaded_data["load"] += 1 / best_target.max_models
                    underloaded_data["memory"] += model.size_gb / best_target.memory_total_gb
                    
                    if len(recommendations) >= 10:  # Limit recommendations
                        break
            
            if len(recommendations) >= 10:
                break
        
        logger.info("Calculated optimal distribution", 
                   moves_recommended=len(recommendations),
                   overloaded_workers=len(overloaded_workers),
                   underloaded_workers=len(underloaded_workers))
        
        return recommendations
    
    def _calculate_move_benefit(self, model: ModelEntry, from_worker: WorkerInfo, 
                              to_worker: WorkerInfo) -> float:
        """Calculate benefit score for moving a model between workers"""
        # Calculate load improvement
        from_load_before = len(from_worker.loaded_models) / from_worker.max_models
        from_load_after = (len(from_worker.loaded_models) - 1) / from_worker.max_models
        
        to_load_before = len(to_worker.loaded_models) / to_worker.max_models  
        to_load_after = (len(to_worker.loaded_models) + 1) / to_worker.max_models
        
        # Calculate memory improvement
        from_memory_before = from_worker.memory_used_gb / from_worker.memory_total_gb
        from_memory_after = (from_worker.memory_used_gb - model.size_gb) / from_worker.memory_total_gb
        
        to_memory_before = to_worker.memory_used_gb / to_worker.memory_total_gb
        to_memory_after = (to_worker.memory_used_gb + model.size_gb) / to_worker.memory_total_gb
        
        # Calculate overall balance improvement
        load_improvement = (from_load_before - from_load_after) - (to_load_after - to_load_before)
        memory_improvement = (from_memory_before - from_memory_after) - (to_memory_after - to_memory_before)
        
        # Weight the improvements
        benefit = load_improvement * 0.6 + memory_improvement * 0.4
        
        return max(0, benefit)


class ClusterOptimizer:
    """Advanced cluster optimization algorithms"""
    
    def __init__(self, load_balancer: WorkerLoadBalancer):
        """Initialize with load balancer"""
        self.load_balancer = load_balancer
        
    def optimize_for_inference_speed(self, workers: List[WorkerInfo], 
                                   models: List[ModelEntry]) -> Dict[str, Any]:
        """Optimize cluster for maximum inference speed"""
        # This could implement more sophisticated algorithms like:
        # - Model co-location for pipeline models
        # - GPU memory bandwidth optimization
        # - Thermal balancing across GPUs
        
        return {
            "strategy": "inference_speed",
            "status": "not_implemented",
            "message": "Advanced optimization algorithms coming in future versions"
        }
    
    def optimize_for_memory_efficiency(self, workers: List[WorkerInfo],
                                     models: List[ModelEntry]) -> Dict[str, Any]:
        """Optimize cluster for memory efficiency"""
        return {
            "strategy": "memory_efficiency", 
            "status": "not_implemented",
            "message": "Memory optimization algorithms coming in future versions"
        }
