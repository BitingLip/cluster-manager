# GPU Cluster Management PowerShell Script
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "test")]
    [string]$Action = "status",
    
    [Parameter(Mandatory=$false)]
    [int]$GpuIndex = -1,
    
    [Parameter(Mandatory=$false)]
    [switch]$All
)

# Configuration
$RedisUrl = "redis://localhost:6379/0"
$ApiPort = 8080
$ApiHost = "localhost"

function Write-Banner {
    param([string]$Text)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Yellow
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "✓ Docker: $dockerVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Docker not found or not running" -ForegroundColor Red
        return $false
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Python not found" -ForegroundColor Red
        return $false
    }
    
    # Check Redis connectivity
    try {
        $redisTest = docker exec gpu-cluster-redis redis-cli ping 2>$null
        if ($redisTest -eq "PONG") {
            Write-Host "✓ Redis: Connected" -ForegroundColor Green
        } else {
            Write-Host "✗ Redis: Not responding" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Redis: Not accessible" -ForegroundColor Red
    }
    
    return $true
}

function Start-Infrastructure {
    Write-Host "Starting Redis infrastructure..." -ForegroundColor Yellow
    docker-compose up -d redis redis-commander flower
    
    Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # Verify services
    $services = @("gpu-cluster-redis", "gpu-cluster-redis-ui", "gpu-cluster-flower")
    foreach ($service in $services) {
        $status = docker ps --filter "name=$service" --format "table {{.Names}}\t{{.Status}}"
        Write-Host "Service $service`: " -NoNewline
        if ($status -match $service) {
            Write-Host "Running" -ForegroundColor Green
        } else {
            Write-Host "Failed" -ForegroundColor Red
        }
    }
}

function Start-ApiGateway {
    Write-Host "Starting API Gateway..." -ForegroundColor Yellow
    
    Push-Location "api_gateway"
    
    # Create .env if it doesn't exist
    if (!(Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from example" -ForegroundColor Green
    }
    
    # Start API in background
    Start-Process -WindowStyle Hidden -FilePath "cmd" -ArgumentList "/c", "start_api.bat"
    
    Pop-Location
    
    # Wait for API to be ready
    Write-Host "Waiting for API Gateway to be ready..." -ForegroundColor Yellow
    $timeout = 30
    $count = 0
    do {
        Start-Sleep -Seconds 2
        $count += 2
        try {
            $response = Invoke-RestMethod -Uri "http://$ApiHost`:$ApiPort/health" -Method Get -ErrorAction SilentlyContinue
            if ($response.status -eq "healthy") {
                Write-Host "✓ API Gateway is ready" -ForegroundColor Green
                return $true
            }
        } catch {
            # Continue waiting
        }
    } while ($count -lt $timeout)
    
    Write-Host "✗ API Gateway failed to start within $timeout seconds" -ForegroundColor Red
    return $false
}

function Start-Worker {
    param([int]$GpuIndex)
    
    Write-Host "Starting Worker for GPU $GpuIndex..." -ForegroundColor Yellow
    
    Push-Location "worker"
    
    # Create .env if it doesn't exist
    if (!(Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
    }
    
    # Update .env with GPU index
    $envContent = Get-Content ".env"
    $envContent = $envContent -replace "^GPU_INDEX=.*", "GPU_INDEX=$GpuIndex"
    $envContent = $envContent -replace "^DIRECTML_DEVICE_INDEX=.*", "DIRECTML_DEVICE_INDEX=$GpuIndex"
    $envContent = $envContent -replace "^WORKER_ID=.*", "WORKER_ID=worker-gpu$GpuIndex-$(Get-Random)"
    $envContent | Set-Content ".env"
    
    # Start worker in background
    $workerId = "worker-gpu$GpuIndex-$(Get-Random)"
    Start-Process -WindowStyle Normal -FilePath "cmd" -ArgumentList "/c", "start_worker.bat $GpuIndex"
    
    Pop-Location
    
    Start-Sleep -Seconds 3
    Write-Host "✓ Worker for GPU $GpuIndex started" -ForegroundColor Green
}

function Get-ClusterStatus {
    Write-Banner "GPU Cluster Status"
    
    # Check infrastructure
    Write-Host "Infrastructure Services:" -ForegroundColor Yellow
    $infraServices = @(
        @{Name="Redis"; Container="gpu-cluster-redis"; Port=6379},
        @{Name="Redis Commander"; Container="gpu-cluster-redis-ui"; Port=8081},
        @{Name="Flower"; Container="gpu-cluster-flower"; Port=5555}
    )
    
    foreach ($service in $infraServices) {
        $status = docker ps --filter "name=$($service.Container)" --format "{{.Status}}" 2>$null
        Write-Host "  $($service.Name): " -NoNewline
        if ($status) {
            Write-Host "Running (Port $($service.Port))" -ForegroundColor Green
        } else {
            Write-Host "Stopped" -ForegroundColor Red
        }
    }
    
    # Check API Gateway
    Write-Host "`nAPI Gateway:" -ForegroundColor Yellow
    try {
        $apiHealth = Invoke-RestMethod -Uri "http://$ApiHost`:$ApiPort/health" -Method Get -ErrorAction Stop
        Write-Host "  Status: Running (Port $ApiPort)" -ForegroundColor Green
        
        # Get cluster stats
        try {
            $clusterStats = Invoke-RestMethod -Uri "http://$ApiHost`:$ApiPort/cluster/status" -Method Get
            Write-Host "  Workers: $($clusterStats.active_workers)/$($clusterStats.total_workers)" -ForegroundColor Green
            Write-Host "  Pending Tasks: $($clusterStats.pending_tasks)" -ForegroundColor Green
            Write-Host "  Running Tasks: $($clusterStats.running_tasks)" -ForegroundColor Green
        } catch {
            Write-Host "  Could not retrieve cluster statistics" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  Status: Stopped or unreachable" -ForegroundColor Red
    }
    
    # Show service URLs
    Write-Host "`nService URLs:" -ForegroundColor Yellow
    Write-Host "  API Gateway: http://localhost:$ApiPort" -ForegroundColor Cyan
    Write-Host "  API Documentation: http://localhost:$ApiPort/docs" -ForegroundColor Cyan
    Write-Host "  Redis Commander: http://localhost:8081" -ForegroundColor Cyan
    Write-Host "  Flower Monitor: http://localhost:5555" -ForegroundColor Cyan
    Write-Host "  Prometheus Metrics: http://localhost:8001/metrics" -ForegroundColor Cyan
}

function Test-Cluster {
    Write-Banner "Testing GPU Cluster"
    
    # Test API Gateway health
    Write-Host "Testing API Gateway..." -ForegroundColor Yellow
    try {
        $health = Invoke-RestMethod -Uri "http://$ApiHost`:$ApiPort/health" -Method Get
        Write-Host "✓ API Gateway health check passed" -ForegroundColor Green
    } catch {
        Write-Host "✗ API Gateway health check failed" -ForegroundColor Red
        return
    }
    
    # Test LLM inference
    Write-Host "`nTesting LLM inference..." -ForegroundColor Yellow
    try {
        $llmRequest = @{
            task_type = "llm"
            model_name = "gpt2"
            payload = @{
                text = "Hello, this is a test"
                max_tokens = 20
            }
        } | ConvertTo-Json -Depth 3
        
        $response = Invoke-RestMethod -Uri "http://$ApiHost`:$ApiPort/submit" -Method Post -Body $llmRequest -ContentType "application/json"
        $taskId = $response.task_id
        Write-Host "✓ LLM task submitted: $taskId" -ForegroundColor Green
        
        # Wait for completion
        $timeout = 60
        $count = 0
        do {
            Start-Sleep -Seconds 2
            $count += 2
            $status = Invoke-RestMethod -Uri "http://$ApiHost`:$ApiPort/status/$taskId" -Method Get
            if ($status.status -eq "success") {
                Write-Host "✓ LLM inference completed successfully" -ForegroundColor Green
                break
            } elseif ($status.status -eq "failure") {
                Write-Host "✗ LLM inference failed" -ForegroundColor Red
                break
            }
        } while ($count -lt $timeout)
        
        if ($count -ge $timeout) {
            Write-Host "✗ LLM inference timed out" -ForegroundColor Red
        }
        
    } catch {
        Write-Host "✗ LLM inference test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Stop-Cluster {
    Write-Banner "Stopping GPU Cluster"
    
    # Stop infrastructure
    Write-Host "Stopping infrastructure services..." -ForegroundColor Yellow
    docker-compose down
    
    # Kill any remaining processes (be careful with this)
    Write-Host "Stopping worker processes..." -ForegroundColor Yellow
    Get-Process | Where-Object {$_.ProcessName -match "python|celery|uvicorn"} | ForEach-Object {
        Write-Host "Stopping process: $($_.ProcessName) (PID: $($_.Id))" -ForegroundColor Yellow
        try {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        } catch {
            # Process may have already stopped
        }
    }
    
    Write-Host "✓ Cluster stopped" -ForegroundColor Green
}

# Main execution
switch ($Action) {
    "start" {
        Write-Banner "Starting GPU Cluster"
        
        if (!(Test-Prerequisites)) {
            Write-Host "Prerequisites check failed. Exiting." -ForegroundColor Red
            exit 1
        }
        
        Start-Infrastructure
        Start-Sleep -Seconds 5
        
        if (Start-ApiGateway) {
            Start-Sleep -Seconds 3
            
            if ($GpuIndex -ge 0) {
                Start-Worker -GpuIndex $GpuIndex
            } elseif ($All) {
                # Start all workers (0-4 for your setup)
                0..4 | ForEach-Object {
                    Start-Worker -GpuIndex $_
                    Start-Sleep -Seconds 2
                }
            } else {
                Write-Host "Specify -GpuIndex or -All to start workers" -ForegroundColor Yellow
            }
        }
        
        Get-ClusterStatus
    }
    
    "stop" {
        Stop-Cluster
    }
    
    "restart" {
        Stop-Cluster
        Start-Sleep -Seconds 5
        & $MyInvocation.MyCommand.Path -Action start -All
    }
    
    "status" {
        Get-ClusterStatus
    }
    
    "test" {
        Test-Cluster
    }
    
    "logs" {
        Write-Banner "Cluster Logs"
        Write-Host "Infrastructure logs:" -ForegroundColor Yellow
        docker-compose logs --tail=50
    }
    
    default {
        Write-Host "Usage: .\manage-cluster.ps1 -Action [start|stop|restart|status|logs|test] [-GpuIndex <n>] [-All]" -ForegroundColor Yellow
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host "  .\manage-cluster.ps1 -Action start -All" -ForegroundColor Cyan
        Write-Host "  .\manage-cluster.ps1 -Action start -GpuIndex 0" -ForegroundColor Cyan
        Write-Host "  .\manage-cluster.ps1 -Action status" -ForegroundColor Cyan
        Write-Host "  .\manage-cluster.ps1 -Action test" -ForegroundColor Cyan
    }
}
