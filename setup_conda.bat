@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ============================================
echo  DAIA v3 - conda environment setup
echo  Python 3.12 / CUDA 12.7
echo ============================================
echo.

:: Check conda
where conda > nul 2>&1
if not %ERRORLEVEL% == 0 (
    echo [ERROR] conda not found.
    echo Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

:: Check if daia_v3 already exists
conda env list | findstr /C:"daia_v3" > nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [INFO] daia_v3 environment already exists.
    set /p OVERWRITE="Reinstall? (y/N): "
    if /i "!OVERWRITE!" == "y" (
        echo Removing existing environment...
        conda env remove -n daia_v3 -y
        if not !ERRORLEVEL! == 0 (
            echo [ERROR] Failed to remove environment.
            pause
            exit /b 1
        )
    ) else (
        goto :ACTIVATE_ONLY
    )
)

:: Create environment
echo.
echo [1/3] Creating conda environment (Python 3.12)...
conda env create -f environment.yml
if not %ERRORLEVEL% == 0 (
    echo [ERROR] Failed to create environment.
    pause
    exit /b 1
)
echo Done.

:: LLM install option
echo.
echo [2/3] llama-cpp-python install option
echo -----------------------------------------
echo  1. CUDA GPU (NVIDIA CUDA 12.7) [Recommended]
echo  2. CPU only
echo  3. Skip
echo -----------------------------------------
set /p LLM_OPT="Choose (1/2/3): "

call conda activate daia_v3

if "!LLM_OPT!" == "1" (
    echo Installing llama-cpp-python with CUDA 12.7...
    set CMAKE_ARGS=-DLLAMA_CUDA=on
    set FORCE_CMAKE=1
    pip install llama-cpp-python --force-reinstall --no-cache-dir
    if not !ERRORLEVEL! == 0 (
        echo [WARN] CUDA build failed. Falling back to CPU...
        pip install llama-cpp-python
    )
)
if "!LLM_OPT!" == "2" (
    pip install llama-cpp-python
)
if "!LLM_OPT!" == "3" (
    echo Skipping LLM install. Rules-based analysis only.
)

goto :DONE

:ACTIVATE_ONLY
echo.
echo Using existing daia_v3 environment.
call conda activate daia_v3

:DONE
echo.
echo [3/3] Setup complete!
echo ============================================
echo  How to run:
echo    conda activate daia_v3
echo    python app.py          (Gradio UI)
echo    python main.py [CSV]   (CLI)
echo ============================================
pause
endlocal
