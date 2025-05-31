@echo off
REM Start GPU Worker for specified GPU index

if "%1"=="" (
    echo Usage: start_worker.bat [GPU_INDEX]
    echo Example: start_worker.bat 0
    exit /b 1
)

set GPU_INDEX=%1
echo Starting GPU Worker for GPU %GPU_INDEX%...

REM Set environment variables
set REDIS_URL=redis://localhost:6379/0
set CELERY_BROKER_URL=redis://localhost:6379/0
set CELERY_RESULT_BACKEND=redis://localhost:6379/0
set GPU_INDEX=%GPU_INDEX%
set DIRECTML_DEVICE_INDEX=%GPU_INDEX%
set WORKER_ID=worker-gpu%GPU_INDEX%-%RANDOM%
set LOG_LEVEL=INFO

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if .env file exists, if not copy from example
REM Ensure a base .env exists, but avoid per-worker modification here
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from example...
        copy .env.example .env
    ) else (
        echo WARNING: .env.example not found, .env file may be missing or incomplete.
    )
)

REM The following lines that modified .env have been removed to prevent race conditions.
REM Settings like GPU_INDEX, DIRECTML_DEVICE_INDEX, and WORKER_ID are now set
REM purely via environment variables above, which pydantic-settings will pick up.

REM Install dependencies if needed
echo Checking dependencies...
pip install -r requirements.txt --quiet

REM Create models directory
if not exist "models" mkdir models

REM Start the worker
echo Starting Celery worker for GPU %GPU_INDEX%...
celery -A app.tasks:celery_app worker --loglevel=info --concurrency=1 --hostname=%WORKER_ID%@%%h --pool=solo

pause
