#!/bin/bash
# DAIA v2 - Linux/Mac 가상환경 설정 스크립트

echo "========================================"
echo " DAIA v2 가상환경 설정 (Linux/Mac)"
echo "========================================"

# Python 버전 확인
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3가 없습니다."
    echo "  Ubuntu: sudo apt install python3 python3-venv"
    echo "  Mac:    brew install python"
    exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYVER"

# 3.10 미만 경고
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if [ $? -ne 0 ]; then
    echo "[WARNING] Python 3.10+ 권장 (현재: $PYVER)"
fi

# 가상환경 생성
echo ""
echo "[1/4] 가상환경 생성 중... (.venv)"
python3 -m venv .venv
echo "  완료"

# 활성화
echo ""
echo "[2/4] 가상환경 활성화"
source .venv/bin/activate

# pip 업그레이드
echo ""
echo "[3/4] pip 업그레이드"
pip install --upgrade pip -q

# 패키지 설치
echo ""
echo "[4/4] 패키지 설치 중..."
pip install -r requirements.txt

# llama-cpp-python 설치 선택
echo ""
echo "========================================"
echo " LLM 설치 옵션 선택"
echo "========================================"
echo " 1. CPU 전용"
echo " 2. CUDA GPU (NVIDIA)"
echo " 3. Metal (Mac Apple Silicon)"
echo " 4. 건너뛰기"
echo "========================================"
read -p "선택 (1/2/3/4): " llm_choice

case $llm_choice in
    1) pip install llama-cpp-python ;;
    2) CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir ;;
    3) CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir ;;
    4) echo "LLM 설치 건너뜀." ;;
    *) echo "잘못된 입력, 건너뜀." ;;
esac

echo ""
echo "========================================"
echo " 설치 완료!"
echo "========================================"
echo ""
echo "사용법:"
echo "  source .venv/bin/activate"
echo "  python main.py data/your_file.csv"
echo "========================================"
