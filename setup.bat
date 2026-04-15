@echo off
:: DAIA v2 - Windows 가상환경 설정 스크립트
echo ========================================
echo  DAIA v2 가상환경 설정 (Windows)
echo ========================================

:: Python 버전 확인
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python이 설치되지 않았습니다.
    echo https://www.python.org/downloads/ 에서 Python 3.10+ 설치 후 재실행하세요.
    pause
    exit /b 1
)

:: 가상환경 생성
echo.
echo [1/4] 가상환경 생성 중... (.venv)
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] 가상환경 생성 실패
    pause
    exit /b 1
)
echo   완료

:: 가상환경 활성화
echo.
echo [2/4] 가상환경 활성화
call .venv\Scripts\activate.bat

:: pip 업그레이드
echo.
echo [3/4] pip 업그레이드
python -m pip install --upgrade pip

:: 기본 패키지 설치
echo.
echo [4/4] 패키지 설치 중...
pip install -r requirements.txt

:: llama-cpp-python 설치 선택
echo.
echo ========================================
echo  LLM 설치 옵션 선택
echo ========================================
echo  1. CPU 전용 (느리지만 범용)
echo  2. CUDA GPU (빠름, NVIDIA GPU 필요)
echo  3. 건너뛰기 (LLM 없이 규칙 기반만 사용)
echo ========================================
set /p llm_choice="선택 (1/2/3): "

if "%llm_choice%"=="1" (
    echo CPU 버전 llama-cpp-python 설치 중...
    pip install llama-cpp-python
)
if "%llm_choice%"=="2" (
    echo CUDA 버전 llama-cpp-python 설치 중...
    set CMAKE_ARGS=-DLLAMA_CUDA=on
    pip install llama-cpp-python --force-reinstall --no-cache-dir
)
if "%llm_choice%"=="3" (
    echo LLM 설치 건너뜀.
)

echo.
echo ========================================
echo  설치 완료!
echo ========================================
echo.
echo 사용법:
echo   .venv\Scripts\activate
echo   python main.py data\your_file.csv
echo.
echo 가상환경 활성화 후 바로 실행하려면:
echo   run.bat data\your_file.csv
echo ========================================
pause
