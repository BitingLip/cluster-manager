# Cluster Manager

🖥️ **Status: ⚙️ Operational**

The Cluster Manager handles GPU cluster orchestration, worker node management, and distributed task processing using Celery and Redis.

## Core Features

- ✅ GPU worker node management
- ✅ Celery-based task distribution  
- ✅ Redis integration for queuing
- ✅ Multi-GPU support
- ✅ Centralized model access
- ✅ Resource monitoring

## Quick Start

```bash
# Start Redis service
docker-compose up -d redis

# Start worker node
cd cluster/worker
python app/worker.py
```

Worker will automatically:
- Connect to Redis queue
- Initialize GPU resources
- Register available task types
- Begin processing tasks

## Architecture

The cluster manager consists of:

- **Redis Server**: Task queue and result storage
- **Celery Workers**: GPU-accelerated task processors  
- **Flower Dashboard**: Worker monitoring (optional)

## Documentation

📖 **Detailed Documentation:**
- [Architecture](docs/architecture.md) - Worker design and GPU management
- [API Reference](docs/api.md) - Task types and interfaces
- [Development](docs/development.md) - Setup and contribution guide
- [Deployment](docs/deployment.md) - Production configuration

## Integration

Works with:
- **Gateway Manager**: Receives task requests
- **Model Manager**: Accesses centralized models
- **Task Manager**: Coordinates task lifecycle

## Configuration

Key environment variables:
- `MODEL_CACHE_DIR`: Path to centralized models
- `REDIS_URL`: Redis connection string  
- `GPU_MEMORY_FRACTION`: GPU memory allocation
- `WORKER_CONCURRENCY`: Tasks per worker

### 1. Start Infrastructure Services
```powershell
docker-compose up -d
```

### 2. Configure Environment
```powershell
copy .env.example .env
# Edit .env with your configuration
```

### 3. Start Workers
```powershell
start_worker.bat
```

### 4. Validate Cluster
```powershell
python validate_cluster.py
```

## Configuration

Environment variables in `.env` (and `.env.example`) are crucial for configuring the cluster behavior:
- `REDIS_URL`: Connection string for the Redis broker.
- `CELERY_RESULT_BACKEND`: Configuration for Celery's result backend.
- `GPU_DEVICES`: Specifies available GPU device IDs for workers.
- `MODEL_CACHE_DIR`: **Note:** While previously used for a local model cache within `cluster-manager`, model storage is now centralized in `model-manager`. This variable might be deprecated or repurposed if workers fetch models directly from a path provided by `model-manager`. Configuration for accessing `model-manager` (e.g., its API endpoint or shared file path) will be important.

## Scripts

- `start_worker.bat`: Start Celery workers
- `manage-cluster.ps1`: PowerShell cluster management utilities
- `validate_cluster.py`: Health check and validation
- `debug_*.py`: Debug utilities for troubleshooting
- `cleanup_debug_files.ps1`: Clean up debug files

## Worker Architecture

Each worker:
1. Registers with the cluster and signals its availability and capabilities (e.g., available GPUs).
2. Receives tasks via the message broker (e.g., Celery/Redis).
3. For tasks requiring AI models, it liaises with the `model-manager` (or uses paths provided by it) to access/load the necessary model files into GPU memory (e.g., using DirectML).
4. Processes inference requests or other computational tasks.
5. Returns results via the message broker or a configured result backend.
6. Reports its health and status.

## Model Management Integration

**Note**: The core logic for model downloading, storage, and detailed management now resides in the `model-manager` submodule. `cluster-manager` interacts with `model-manager` to ensure workers have access to the required models for their tasks. This might involve API calls to `model-manager` to get model paths or direct access to a shared model storage location managed by `model-manager`. The `app/models/` directory previously within `cluster-manager` has been removed to centralize model-related Python modules in `model-manager`.

## Future Separation

**Note**: Currently contains worker code that will eventually be separated into the `model-manager` component when the entanglement is resolved. The separation will involve:
- Moving model loading logic to model-manager
- Creating clean APIs between cluster-manager and model-manager
- Maintaining worker orchestration in cluster-manager
- Separating model lifecycle management

## Dependencies

See `requirements.txt` for Python dependencies. Key packages:
- `celery[redis]`: Distributed task queue
- `torch-directml`: AMD GPU support for PyTorch
- `transformers`: HuggingFace model support
- `redis`: Redis client
- `psutil`: System monitoring