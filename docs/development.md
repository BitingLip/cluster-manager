# Cluster Manager Development Guide

## Development Environment Setup

### Prerequisites
- Python 3.10+
- Redis server
- ROCm drivers (for AMD GPU support)
- Git

### Initial Setup

1. **Clone and enter the module:**
   ```bash
   cd cluster-manager
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start Redis server:**
   ```bash
   redis-server
   ```

6. **Run the development server:**
   ```bash
   python app/main.py
   ```

### Environment Configuration

Key environment variables in `.env`:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-development-api-key

# Cluster Settings
MAX_WORKERS=10
TASK_TIMEOUT=300
HEARTBEAT_INTERVAL=30
WORKER_REGISTRATION_TIMEOUT=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# GPU Settings
GPU_MEMORY_THRESHOLD=0.8
MODEL_CACHE_SIZE=5
AUTO_SCALE_ENABLED=true
```

## Project Structure

```
cluster-manager/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── models/              # Data models and schemas
│   │   ├── __init__.py
│   │   ├── worker.py        # Worker-related models
│   │   ├── task.py          # Task-related models
│   │   └── cluster.py       # Cluster-related models
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   ├── worker_manager.py    # Worker lifecycle management
│   │   ├── task_scheduler.py    # Task scheduling logic
│   │   ├── load_balancer.py     # Load balancing algorithms
│   │   └── cluster_monitor.py   # Cluster monitoring
│   ├── api/                 # API route handlers
│   │   ├── __init__.py
│   │   ├── workers.py       # Worker management endpoints
│   │   ├── tasks.py         # Task management endpoints
│   │   ├── cluster.py       # Cluster operation endpoints
│   │   └── health.py        # Health check endpoints
│   ├── workers/             # Celery worker definitions
│   │   ├── __init__.py
│   │   ├── inference.py     # Inference task workers
│   │   └── management.py    # Management task workers
│   └── utils/               # Utility modules
│       ├── __init__.py
│       ├── logging.py       # Logging configuration
│       ├── gpu.py           # GPU utilities
│       └── config.py        # Configuration management
├── tests/                   # Test files
├── docs/                    # Detailed documentation
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
└── README.md               # Minimal overview
```

## Development Workflow

### 1. Feature Development

1. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement changes:**
   - Add/modify code in appropriate modules
   - Follow coding standards (see below)
   - Add tests for new functionality

3. **Test your changes:**
   ```bash
   # Run unit tests
   pytest tests/

   # Run integration tests
   pytest tests/integration/

   # Run specific test file
   pytest tests/test_worker_manager.py -v
   ```

4. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: add load balancing algorithm"
   git push origin feature/your-feature-name
   ```

### 2. Testing

#### Unit Tests
```bash
# Run all unit tests
pytest tests/unit/

# Run with coverage
pytest tests/unit/ --cov=app --cov-report=html

# Run specific test class
pytest tests/unit/test_worker_manager.py::TestWorkerManager
```

#### Integration Tests
```bash
# Start test environment (Redis, etc.)
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/

# Cleanup
docker-compose -f docker-compose.test.yml down
```

#### Load Tests
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load/test_api_endpoints.py --host=http://localhost:8000
```

### 3. Debugging

#### FastAPI Debug Mode
```python
# In app/main.py, enable debug mode
app = FastAPI(debug=True)

# Add debug logs
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Celery Worker Debugging
```bash
# Start worker with debug logging
celery -A app.workers worker --loglevel=debug

# Monitor task execution
celery -A app.workers flower
```

#### GPU Debugging
```bash
# Check GPU status
rocm-smi

# Monitor GPU usage during development
watch -n 1 rocm-smi
```

## Coding Standards

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Maximum line length: 88 characters
- Use black for code formatting

```bash
# Format code
black app/ tests/

# Check code style
flake8 app/ tests/

# Sort imports
isort app/ tests/
```

### Documentation
- Use docstrings for all classes and functions
- Follow Google docstring format
- Include type information in docstrings

```python
def schedule_task(task: Task, workers: List[Worker]) -> Optional[Worker]:
    """Schedule a task to the optimal worker.
    
    Args:
        task: The task to schedule
        workers: List of available workers
        
    Returns:
        The selected worker, or None if no worker is available
        
    Raises:
        SchedulingError: If task cannot be scheduled
    """
    pass
```

### Error Handling
- Use custom exception classes
- Log errors with appropriate levels
- Provide meaningful error messages

```python
class WorkerNotFoundError(Exception):
    """Raised when a worker cannot be found."""
    pass

def get_worker(worker_id: str) -> Worker:
    try:
        worker = find_worker(worker_id)
        if not worker:
            raise WorkerNotFoundError(f"Worker {worker_id} not found")
        return worker
    except Exception as e:
        logger.error(f"Failed to get worker {worker_id}: {e}")
        raise
```

## Performance Optimization

### Database Queries
- Use connection pooling for Redis
- Implement caching for frequently accessed data
- Monitor query performance

### Memory Management
- Monitor memory usage of workers
- Implement proper cleanup in event handlers
- Use memory profiling tools

```bash
# Profile memory usage
python -m memory_profiler app/main.py

# Monitor Redis memory
redis-cli info memory
```

### Async Programming
- Use async/await for I/O operations
- Avoid blocking operations in async contexts
- Use proper connection management

## Monitoring and Logging

### Structured Logging
```python
import structlog

logger = structlog.get_logger(__name__)

logger.info(
    "task_scheduled",
    task_id=task.id,
    worker_id=worker.id,
    estimated_duration=task.estimated_duration
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

TASK_COUNTER = Counter('tasks_total', 'Total tasks', ['status'])
TASK_DURATION = Histogram('task_duration_seconds', 'Task duration')

# In your code
TASK_COUNTER.labels(status='completed').inc()
TASK_DURATION.observe(task.duration)
```

### Health Checks
- Implement comprehensive health checks
- Monitor external dependencies
- Set up alerting for critical issues

## Security Considerations

### API Security
- Validate all input parameters
- Use rate limiting
- Implement proper authentication

### Worker Security
- Validate worker registration
- Use secure communication channels
- Implement resource quotas

### Data Protection
- Encrypt sensitive configuration
- Sanitize log outputs
- Implement audit logging

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Check configuration
   redis-cli config get bind
   ```

2. **Worker Registration Failures**
   ```bash
   # Check worker logs
   tail -f logs/worker.log
   
   # Verify network connectivity
   telnet cluster-manager-host 8000
   ```

3. **GPU Memory Issues**
   ```bash
   # Check GPU memory
   rocm-smi --showmeminfo
   
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### Debug Commands
```bash
# Check cluster status
curl -X GET http://localhost:8000/cluster/status

# List workers
curl -X GET http://localhost:8000/workers

# Check task queue
celery -A app.workers inspect active

# Monitor system resources
htop
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Pull Request Guidelines
- Include clear description of changes
- Reference related issues
- Include test coverage
- Update documentation as needed
