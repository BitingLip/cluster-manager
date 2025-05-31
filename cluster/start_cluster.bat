@echo off
REM Start complete GPU cluster (Redis + API Gateway + All Workers)

echo ========================================
echo     GPU Cluster Startup Script
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

REM Start Redis infrastructure
echo Starting Redis infrastructure...
docker-compose up -d redis redis-commander flower
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Redis infrastructure
    pause
    exit /b 1
)

echo Waiting for Redis to be ready...
timeout /t 5 /nobreak >nul

REM Start API Gateway
echo.
echo Starting API Gateway...
start "API Gateway" cmd /k "cd api_gateway && start_api.bat"

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

REM Start Workers for each GPU
echo.
echo Starting GPU Workers...

REM Worker for GPU 0 (RX 6800)
start "GPU Worker 0" cmd /k "cd worker && start_worker.bat 0"
timeout /t 2 /nobreak >nul

REM Worker for GPU 1 (RX 6800)
start "GPU Worker 1" cmd /k "cd worker && start_worker.bat 1"
timeout /t 2 /nobreak >nul

REM Worker for GPU 2 (RX 6800)
start "GPU Worker 2" cmd /k "cd worker && start_worker.bat 2"
timeout /t 2 /nobreak >nul

REM Worker for GPU 3 (RX 6800)
start "GPU Worker 3" cmd /k "cd worker && start_worker.bat 3"
timeout /t 2 /nobreak >nul

REM Worker for GPU 4 (RX 6800 XT)
start "GPU Worker 4" cmd /k "cd worker && start_worker.bat 4"

echo.
echo ========================================
echo     GPU Cluster Started Successfully!
echo ========================================
echo.
echo Services:
echo - API Gateway: http://localhost:8080
echo - API Documentation: http://localhost:8080/docs
echo - Redis Commander: http://localhost:8081
echo - Flower (Celery Monitor): http://localhost:5555
echo - Prometheus Metrics: http://localhost:8001/metrics
echo.
echo GPU Workers:
echo - GPU 0 (RX 6800): Running
echo - GPU 1 (RX 6800): Running
echo - GPU 2 (RX 6800): Running
echo - GPU 3 (RX 6800): Running
echo - GPU 4 (RX 6800 XT): Running
echo.
echo Press any key to view cluster status...
pause >nul

REM Open browser to API documentation
start http://localhost:8080/docs

echo Cluster startup complete!
pause
