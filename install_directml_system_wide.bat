@echo off
echo DirectML System-Wide Installation Script
echo ==========================================
echo.

REM Critical DirectML Requirements Check
echo [STEP 1] Checking DirectML Requirements...
echo.
echo CRITICAL REQUIREMENTS:
echo 1. AMD Adrenalin Edition 23.40.27.06 driver installed
echo 2. NO virtual environments active
echo 3. System Python installation available
echo.

REM Check if in virtual environment
if defined VIRTUAL_ENV (
    echo ERROR: Virtual environment detected: %VIRTUAL_ENV%
    echo DirectML REQUIRES system-wide Python installation
    echo Please run: deactivate
    echo Then re-run this script
    pause
    exit /b 1
)

echo [STEP 2] Installing AMD DirectML packages (system-wide)...
echo.
python -m pip install --upgrade pip
if errorlevel 1 goto error

python -m pip install torch torchvision torchaudio
if errorlevel 1 goto error

python -m pip install torch-directml
if errorlevel 1 goto error

python -m pip install onnxruntime-directml
if errorlevel 1 goto error

python -m pip install numpy pillow opencv-python
if errorlevel 1 goto error

echo.
echo [STEP 3] Verifying DirectML installation...
echo.
python -c "import torch_directml; print(f'DirectML devices: {torch_directml.device_count()}')"
if errorlevel 1 goto error

python -c "import torch_directml; device = torch_directml.device(); print(f'DirectML device: {device}')"
if errorlevel 1 goto error

echo.
echo ==========================================
echo DirectML Installation Completed Successfully!
echo ==========================================
echo.
echo IMPORTANT USAGE NOTES:
echo 1. Always use system Python for DirectML workloads
echo 2. NEVER activate virtual environments for AMD GPU work
echo 3. Virtual environments will break DirectML functionality
echo.
echo Test DirectML with:
echo python -c "import torch_directml; print('DirectML ready!')"
echo.
pause
exit /b 0

:error
echo.
echo ERROR: Installation failed!
echo Check the error messages above
echo.
pause
exit /b 1
