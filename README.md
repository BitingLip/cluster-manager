# Cluster Manager

🖥️ **Status: ✅ Production Ready**

The Cluster Manager provides distributed cluster orchestration with PostgreSQL persistence, real-time resource monitoring, and node lifecycle management for the BitingLip platform.

## Core Features

- ✅ **Real-time Monitoring**: 30-second system resource tracking (CPU, Memory, Disk)
- ✅ **Database Persistence**: PostgreSQL integration with connection pooling
- ✅ **Node Management**: Automatic registration, heartbeat monitoring, cleanup
- ✅ **Cluster Health**: Comprehensive health metrics and analytics
- ✅ **Production Ready**: Robust error handling and transaction management
- ✅ **GPU Detection**: Automatic GPU resource discovery and monitoring

## Quick Start

```bash
# Start the cluster manager
python start_cluster.py

# Check cluster status
python utils/check_cluster_data.py

# Debug cluster components
python utils/debug_cluster.py
```

The cluster manager will automatically:

- Initialize PostgreSQL database schema
- Register itself as master node
- Begin continuous monitoring loop
- Track system resources every 30 seconds
- Cleanup stale nodes automatically

## Architecture

The cluster manager consists of:

- **Database Layer**: PostgreSQL with connection pooling for persistence
- **Monitoring Engine**: Real-time system resource collection using psutil
- **Node Registry**: Automatic node discovery and lifecycle management
- **Health Analytics**: Cluster-wide metrics and health monitoring

## Project Structure

```
cluster-manager/
├── app/                    # Core application code
│   ├── cluster_manager.py  # Main cluster manager class
│   ├── database.py         # PostgreSQL database operations
│   └── simple_config.py    # Configuration management
├── config/                 # Configuration files
│   └── cluster-manager-db.env
├── utils/                  # Utility scripts
│   ├── check_cluster_data.py   # Database status checker
│   ├── debug_cluster.py        # Component testing
│   └── validate_cluster.py     # Cluster validation
├── docs/                   # Documentation
└── start_cluster.py        # Main startup script
```

## Documentation

📖 **Detailed Documentation:**

- [Architecture](docs/architecture.md) - Worker design and GPU management
- [API Reference](docs/api.md) - Task types and interfaces
- [Development](docs/development.md) - Setup and contribution guide
- [Deployment](docs/deployment.md) - Production configuration

## Configuration

Configure via environment variables in `config/cluster-manager-db.env`:

```bash
# Database Configuration
CLUSTER_DB_HOST=localhost
CLUSTER_DB_PORT=5432
CLUSTER_DB_NAME=bitinglip_cluster
CLUSTER_DB_USER=postgres
CLUSTER_DB_PASSWORD=password

# Cluster Manager Configuration
CLUSTER_MANAGER_HOST=localhost
CLUSTER_MANAGER_PORT=8002
DEBUG=true
```

## Database Schema

The cluster manager creates and manages these PostgreSQL tables:

- **cluster_nodes**: Node registration and status tracking
- **system_resources**: Real-time resource metrics (CPU, Memory, Disk)
- **cluster_events**: Event logging (optional)
- **gpu_resources**: GPU metrics and information (optional)

Schema is automatically initialized on first startup via `database.initialize_schema()`.

## Live Monitoring

Once running, the cluster manager provides:

- **Real-time Metrics**: Updated every 30 seconds
- **Node Health**: Automatic heartbeat monitoring
- **Resource Tracking**: CPU, Memory, Disk usage trends
- **Cluster Analytics**: Aggregate health and performance metrics

Example live data:

```
📍 cluster_manager_DESKTOP-2TQL1QP_20250603_123715
   Type: master
   Status: active
   CPU: 67.7% | Memory: 43.9% | Disk: 47.1%
   Last Heartbeat: 2025-06-03 12:37:46
```

## Integration

The cluster manager integrates with:

- **Model Manager**: Node registration for model serving
- **Gateway Manager**: Cluster status and health reporting
- **Task Manager**: Resource availability and load balancing
- **PostgreSQL**: Persistent storage for all cluster data

## Utilities

- `utils/check_cluster_data.py`: View current cluster status and metrics
- `utils/debug_cluster.py`: Test all components and run diagnostics
- `utils/validate_cluster.py`: Comprehensive cluster validation

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
