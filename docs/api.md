# Cluster Manager API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
All API endpoints require authentication via API key in the header:
```
Authorization: Bearer <api_key>
```

## Endpoints

### Health & Status

#### GET `/health`
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-05-30T10:00:00Z",
  "version": "1.0.0"
}
```

#### GET `/cluster/status`
Comprehensive cluster status information.

**Response:**
```json
{
  "cluster_id": "biting-lip-cluster",
  "status": "operational",
  "total_workers": 5,
  "active_workers": 4,
  "queued_tasks": 12,
  "running_tasks": 8,
  "gpu_utilization": 65.5,
  "total_memory": "40GB",
  "available_memory": "14GB",
  "uptime": "7d 4h 23m"
}
```

### Worker Management

#### GET `/workers`
List all registered workers and their status.

**Query Parameters:**
- `status` (optional): Filter by worker status (`online`, `offline`, `busy`, `error`)
- `gpu_type` (optional): Filter by GPU type

**Response:**
```json
{
  "workers": [
    {
      "worker_id": "worker-001",
      "status": "online",
      "gpu_info": {
        "model": "AMD Radeon RX 7900 XTX",
        "memory_total": "24GB",
        "memory_used": "8GB",
        "temperature": 65,
        "utilization": 75.5
      },
      "loaded_models": ["llama-2-7b", "stable-diffusion-xl"],
      "current_tasks": 2,
      "last_heartbeat": "2025-05-30T10:00:00Z",
      "performance_score": 8.5
    }
  ],
  "total": 5,
  "online": 4,
  "offline": 1
}
```

#### GET `/workers/{worker_id}`
Get detailed information about a specific worker.

**Path Parameters:**
- `worker_id`: Unique identifier for the worker

**Response:**
```json
{
  "worker_id": "worker-001",
  "status": "online",
  "gpu_info": {
    "model": "AMD Radeon RX 7900 XTX",
    "memory_total": "24GB",
    "memory_used": "8GB",
    "temperature": 65,
    "utilization": 75.5,
    "driver_version": "ROCm 5.7.0"
  },
  "system_info": {
    "cpu": "AMD Ryzen 9 7950X",
    "ram_total": "64GB",
    "ram_used": "12GB",
    "os": "Ubuntu 22.04"
  },
  "loaded_models": [
    {
      "name": "llama-2-7b",
      "memory_usage": "6.5GB",
      "load_time": "45s",
      "last_used": "2025-05-30T09:55:00Z"
    }
  ],
  "task_history": {
    "completed_today": 45,
    "average_duration": "2.3s",
    "success_rate": 98.5
  },
  "metrics": {
    "tasks_per_hour": 120,
    "average_response_time": "1.8s",
    "error_rate": 0.02
  }
}
```

#### POST `/workers/{worker_id}/restart`
Restart a specific worker.

**Path Parameters:**
- `worker_id`: Unique identifier for the worker

**Response:**
```json
{
  "message": "Worker restart initiated",
  "worker_id": "worker-001",
  "estimated_downtime": "30s"
}
```

### Task Management

#### POST `/tasks`
Submit a new task to the cluster.

**Request Body:**
```json
{
  "task_type": "text_generation",
  "model_name": "llama-2-7b",
  "parameters": {
    "prompt": "Explain quantum computing",
    "max_tokens": 512,
    "temperature": 0.7
  },
  "priority": "normal",
  "timeout": 300
}
```

**Response:**
```json
{
  "task_id": "task-12345",
  "status": "queued",
  "estimated_wait_time": "15s",
  "assigned_worker": null,
  "created_at": "2025-05-30T10:00:00Z"
}
```

#### GET `/tasks/{task_id}`
Get status and results of a specific task.

**Path Parameters:**
- `task_id`: Unique identifier for the task

**Response:**
```json
{
  "task_id": "task-12345",
  "status": "completed",
  "assigned_worker": "worker-001",
  "created_at": "2025-05-30T10:00:00Z",
  "started_at": "2025-05-30T10:00:15Z",
  "completed_at": "2025-05-30T10:00:18Z",
  "duration": "3.2s",
  "result": {
    "generated_text": "Quantum computing is a revolutionary...",
    "tokens_generated": 324,
    "processing_time": "2.8s"
  }
}
```

#### GET `/tasks`
List tasks with filtering and pagination.

**Query Parameters:**
- `status` (optional): Filter by task status
- `worker_id` (optional): Filter by assigned worker
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "tasks": [
    {
      "task_id": "task-12345",
      "status": "completed",
      "task_type": "text_generation",
      "assigned_worker": "worker-001",
      "duration": "3.2s",
      "created_at": "2025-05-30T10:00:00Z"
    }
  ],
  "total": 1250,
  "limit": 50,
  "offset": 0
}
```

#### DELETE `/tasks/{task_id}`
Cancel a queued or running task.

**Path Parameters:**
- `task_id`: Unique identifier for the task

**Response:**
```json
{
  "message": "Task cancelled",
  "task_id": "task-12345",
  "status": "cancelled"
}
```

### Model Management

#### POST `/models/assign`
Assign a model to specific workers.

**Request Body:**
```json
{
  "model_name": "llama-2-7b",
  "worker_ids": ["worker-001", "worker-002"],
  "preload": true,
  "priority": "high"
}
```

**Response:**
```json
{
  "assignment_id": "assign-789",
  "model_name": "llama-2-7b",
  "assignments": [
    {
      "worker_id": "worker-001",
      "status": "loading",
      "estimated_time": "45s"
    },
    {
      "worker_id": "worker-002",
      "status": "queued",
      "estimated_time": "90s"
    }
  ]
}
```

#### POST `/models/unload`
Unload a model from workers.

**Request Body:**
```json
{
  "model_name": "llama-2-7b",
  "worker_ids": ["worker-001"],
  "force": false
}
```

**Response:**
```json
{
  "message": "Model unload initiated",
  "model_name": "llama-2-7b",
  "affected_workers": ["worker-001"]
}
```

#### GET `/models/assignments`
Get current model assignments across workers.

**Response:**
```json
{
  "assignments": [
    {
      "worker_id": "worker-001",
      "models": [
        {
          "name": "llama-2-7b",
          "status": "loaded",
          "memory_usage": "6.5GB",
          "load_time": "45s"
        }
      ]
    }
  ]
}
```

### Cluster Operations

#### POST `/cluster/rebalance`
Trigger cluster rebalancing to optimize resource distribution.

**Request Body:**
```json
{
  "strategy": "memory_optimized",
  "force": false,
  "dry_run": false
}
```

**Response:**
```json
{
  "rebalance_id": "rebal-456",
  "status": "initiated",
  "estimated_duration": "2m",
  "affected_workers": ["worker-001", "worker-003"],
  "operations": [
    {
      "type": "model_migration",
      "from_worker": "worker-001",
      "to_worker": "worker-003",
      "model": "stable-diffusion-xl"
    }
  ]
}
```

#### GET `/cluster/metrics`
Get cluster performance metrics.

**Query Parameters:**
- `timeframe` (optional): Time range for metrics (`1h`, `24h`, `7d`, `30d`)

**Response:**
```json
{
  "timeframe": "24h",
  "metrics": {
    "total_tasks": 2450,
    "successful_tasks": 2398,
    "failed_tasks": 52,
    "success_rate": 97.9,
    "average_response_time": "2.1s",
    "peak_throughput": "150 tasks/min",
    "gpu_utilization": {
      "average": 68.5,
      "peak": 95.2,
      "minimum": 12.1
    },
    "memory_usage": {
      "average": "75%",
      "peak": "92%",
      "efficiency": 85.3
    }
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource conflict |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Cluster overloaded |

## Rate Limits

- Default: 1000 requests per hour per API key
- Burst: 100 requests per minute
- WebSocket connections: 10 concurrent per client

## WebSocket Events

Connect to `/ws` for real-time cluster events:

### Events
- `worker_status_changed`
- `task_completed`
- `task_failed`
- `cluster_rebalance_started`
- `model_loaded`
- `gpu_alert`

### Example Event:
```json
{
  "event": "task_completed",
  "timestamp": "2025-05-30T10:00:18Z",
  "data": {
    "task_id": "task-12345",
    "worker_id": "worker-001",
    "duration": "3.2s",
    "success": true
  }
}
```
