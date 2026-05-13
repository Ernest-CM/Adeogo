@echo off
REM ==========================================
REM Setup Script for Plant Disease Detection
REM Run this from Conda Prompt (NOT regular cmd.exe)
REM ==========================================

echo [PHASE 0] Setting up conda environment...
echo.

REM Activate the adonye environment
call conda activate adonye
if errorlevel 1 (
    echo ERROR: Could not activate adonye environment.
    echo Make sure you are running this from the Anaconda Prompt or Miniconda Prompt.
    pause
    exit /b 1
)

echo [Step 1/4] Uninstalling tensorflow_cpu...
pip uninstall tensorflow_cpu -y

echo.
echo [Step 2/4] Installing GPU TensorFlow (tensorflow[and-cuda]==2.19.1)...
REM Note: This will download CUDA/cuDNN wheels, ~1-2 GB
pip install "tensorflow[and-cuda]==2.19.1"

echo.
echo [Step 3/4] Installing missing packages (opencv, kaggle, jupyter, ipykernel)...
pip install "opencv-python==4.10.0.84" "kaggle>=1.6" "jupyter>=1.1" "ipykernel>=6.29"

echo.
echo [Step 4/4] Installing ipykernel for Jupyter...
python -m ipykernel install --user --name adonye --display-name "Python (adonye)"

echo.
echo ==========================================
echo [VERIFICATION] Testing imports...
python -c "import tensorflow as tf, cv2, streamlit, kaggle, jupyter, sklearn; print('TensorFlow version:', tf.__version__); print('OpenCV version:', cv2.__version__); print('All imports successful!')"

if errorlevel 1 (
    echo ERROR: Some imports failed. Check installation.
    pause
    exit /b 1
)

echo.
echo [NEXT STEPS]
echo 1. Setup Kaggle credentials: https://www.kaggle.com/account/api
echo 2. Download kaggle.json and place at: C:\Users\User\.kaggle\kaggle.json
echo 3. Run Phase 1: python src\config.py
echo.
echo Setup complete!
pause
