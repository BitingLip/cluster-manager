@echo off
echo Activating AMD GPU Environment...
call gpu_environments\amd_gpu_env\Scripts\activate.bat
echo Environment activated. DirectML-optimized packages available.
echo AMD GPUs detected: 2
set TF_DIRECTML_DEVICE_COUNT=2
echo Ready for AMD GPU workloads!
cmd /k
