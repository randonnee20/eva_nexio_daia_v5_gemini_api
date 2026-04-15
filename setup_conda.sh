#!/bin/bash
# DAIA v3 - conda environment setup (Linux/Mac)
# Python 3.12 / CUDA 12.7

set -e

echo "============================================"
echo " DAIA v3 - conda environment setup"
echo " Python 3.12"
echo "============================================"
echo

if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found."
    echo "Install Anaconda or Miniconda first."
    exit 1
fi

# Check existing env
if conda env list | grep -q "daia_v3"; then
    echo "[INFO] daia_v3 already exists."
    read -p "Reinstall? (y/N): " OVERWRITE
    if [[ "$OVERWRITE" =~ ^[Yy]$ ]]; then
        conda env remove -n daia_v3 -y
    else
        echo "conda activate daia_v3 && python app.py"
        exit 0
    fi
fi

echo "[1/3] Creating conda environment (Python 3.12)..."
conda env create -f environment.yml
echo "Done."

echo
echo "[2/3] llama-cpp-python install option"
echo "  1. CUDA GPU (NVIDIA)"
echo "  2. Metal (Mac Apple Silicon)"
echo "  3. CPU only"
echo "  4. Skip"
read -p "Choose (1/2/3/4): " LLM_OPT

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate daia_v3

case $LLM_OPT in
    1) CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 \
       pip install llama-cpp-python --force-reinstall --no-cache-dir ;;
    2) CMAKE_ARGS="-DLLAMA_METAL=on" \
       pip install llama-cpp-python --force-reinstall --no-cache-dir ;;
    3) pip install llama-cpp-python ;;
    *) echo "Skipping LLM install." ;;
esac

echo
echo "[3/3] Setup complete!"
echo "============================================"
echo "  conda activate daia_v3"
echo "  python app.py     # Gradio UI -> http://localhost:7860"
echo "  python main.py [CSV]  # CLI"
echo "============================================"
