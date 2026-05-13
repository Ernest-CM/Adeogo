# Implementation Guide: Plant Disease Detection System

**Project:** Real-Time Diseases Detection in Solanaceous Crops Using CNN  
**Author:** Ekunjesu Adeogo (22CD009343)  
**Institution:** Landmark University, Department of Computer Science  
**Status:** In Development (Phases 0-3 scaffolding complete)

---

## ✅ What's Been Created

### Core Project Files
- ✅ `src/config.py` — All hyperparameters, paths, disease info database
- ✅ `src/__init__.py` — Package initialization
- ✅ `app/__init__.py` — App package initialization
- ✅ `src/data_prep.py` — Dataset download, filter, split logic
- ✅ `src/augmentation.py` — TensorFlow augmentation pipeline
- ✅ `src/model.py` — MobileNetV2 transfer learning model
- ✅ `environment.yml` — Conda environment specification
- ✅ `requirements.txt` — pip requirements fallback
- ✅ `SETUP_CONDA.bat` — Windows batch setup script
- ✅ Directory structure (`data/`, `models/`, `notebooks/`, `src/`, `app/`)

### Project Documents
- ✅ `project_document.txt` — Original project document (Chapters 1-3)
- ✅ `revised.md` — Corrected & completed document (Chapters 1-5, with fixes)

---

## 🚀 Next Steps (Immediate)

### Phase 0: Environment Setup (Required before anything else)

**Run in Anaconda Prompt (NOT regular cmd.exe):**

```bash
cd c:\Users\User\Desktop\PROJECTS\Adeogo\Adeogo
SETUP_CONDA.bat
```

OR manually run:

```bash
conda activate adonye
pip uninstall tensorflow_cpu -y
pip install "tensorflow[and-cuda]==2.19.1" "opencv-python==4.10.0.84" "kaggle>=1.6" "jupyter>=1.1" "ipykernel>=6.29"
python -m ipykernel install --user --name adonye --display-name "Python (adonye)"
```

**Verification:**
```bash
python -c "import tensorflow as tf, cv2, kaggle; print('TensorFlow:', tf.__version__); print('OpenCV:', cv2.__version__)"
```

---

### Phase 0.5: Kaggle Setup (Required for dataset download)

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" → downloads `kaggle.json`
3. Create folder: `C:\Users\User\.kaggle\` (if it doesn't exist)
4. Move `kaggle.json` → `C:\Users\User\.kaggle\kaggle.json`

**Verification:**
```bash
kaggle datasets list
```

---

### Phase 1-2: Download & Prepare Dataset

Once environment is set up:

```bash
conda activate adonye
cd c:\Users\User\Desktop\PROJECTS\Adeogo\Adeogo
python src/data_prep.py
```

This will:
- Download PlantVillage dataset from Kaggle (~1.4 GB, takes 5-10 min)
- Filter to 15 solanaceous classes (Tomato 10, Potato 3, Pepper 2)
- Split into 70% train / 15% val / 15% test
- Save class names to `models/class_names.json`

**Expected output structure after this step:**
```
data/processed/
├── train/          (15,950 images)
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── ... (15 classes total)
├── val/           (3,420 images)
│   └── [same 15 classes]
└── test/          (3,420 images)
    └── [same 15 classes]
```

---

## 📋 Complete Phase Breakdown

| Phase | Task | Effort | Status |
|-------|------|--------|--------|
| 0 | Environment setup + Kaggle | 30 min | 🔴 TODO |
| 1 | Directory structure | 10 min | ✅ Done |
| 2 | Dataset download + split | 60-90 min | 🟡 Code ready, needs Phase 0 |
| 3a | Write training pipeline | 60 min | 🟡 Partial (augmentation.py, model.py done) |
| 3b | **Train on GPU/Colab** | 2-3 hrs | 🔴 TODO |
| 4 | Evaluation + metrics | 45 min | 🔴 TODO |
| 5 | Inference utility | 30 min | 🔴 TODO |
| 6 | Real-time webcam | 45 min | 🔴 TODO |
| 7 | Streamlit web app | 60 min | 🔴 TODO |
| 8 | TFLite export | 20 min | 🔴 TODO |

---

## 🎯 Recommended Execution Order

### Day 1 (Setup Phase 0-2)
1. Run `SETUP_CONDA.bat` in Anaconda Prompt
2. Setup kaggle.json
3. Run `python src/data_prep.py` — dataset will download & prepare

### Day 2 (Training Phase 3)
- Write `src/train.py` (partially complete, need full training loop)
- Run training: `python src/train.py` on GPU A2000 (~2-3 hours)
- Alternative: Use Google Colab (free T4 GPU, ~60 min)

### Day 3-4 (Evaluation & App Phases 4-7)
- `src/evaluate.py` — test metrics
- `src/predict.py` — inference utility
- `src/realtime.py` — webcam demo
- `app/streamlit_app.py` — web UI

### Day 5 (Final Phase 8 + Verification)
- TFLite export
- Full system test
- Update `revised.md` with actual results

---

## 📁 Files Still Needed

### Remaining Source Code (4 files)
- `src/train.py` — Training loop, callbacks, epoch tracking
- `src/evaluate.py` — Metrics, confusion matrix, per-class analysis
- `src/predict.py` — Single-image inference with model caching
- `src/realtime.py` — OpenCV webcam inference + overlay

### Remaining App (1 file)
- `app/streamlit_app.py` — Web upload + prediction UI

### Remaining Notebooks (3 files)
- `notebooks/01_data_exploration.ipynb` — Dataset analysis
- `notebooks/02_model_training.ipynb` — Training workflow
- `notebooks/03_evaluation.ipynb` — Results visualization

---

## ⚠️ Critical Windows Notes

1. **Always use Conda Prompt** (not regular cmd.exe) for conda commands
2. **OpenCV on Windows:** `cv2.VideoCapture(0, cv2.CAP_DSHOW)` avoids 30-second freeze
3. **Paths:** Use `pathlib.Path` objects in Python, not hardcoded backslashes
4. **GPU TensorFlow:** May require uninstall/reinstall of `tensorflow_cpu` → `tensorflow[and-cuda]`
5. **Keras 3:** Use modern `image_dataset_from_directory` API, not deprecated `ImageDataGenerator`

---

## 📊 Expected Results (from project document)

After completing all phases, the system should achieve:

| Metric | Target |
|--------|--------|
| Test Accuracy | 96.1% |
| Precision | 95.8% |
| Recall | 95.4% |
| F1-Score | 95.6% |
| Real-time FPS | ~11.5 FPS |
| Model Size | 15-20 MB (Keras) |
| TFLite Size | ~4-5 MB |

---

## 🔧 Troubleshooting

### `kaggle: command not found`
- Make sure you're in an Anaconda Prompt (not cmd.exe or PowerShell)
- Run: `pip install kaggle`

### `ModuleNotFoundError: No module named 'tensorflow'`
- Conda environment not activated
- Run: `conda activate adonye` first

### `tensorflow_cpu` vs GPU TensorFlow
- Current env has CPU-only TF
- To use A2000 GPU: `pip uninstall tensorflow_cpu -y && pip install tensorflow[and-cuda]==2.19.1`
- If GPU TF fails, use Google Colab instead

### Dataset download fails
- Check kaggle.json exists at `C:\Users\User\.kaggle\kaggle.json`
- Run: `kaggle datasets list` to verify setup
- Alternative: Download manually from https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### Webcam not found in realtime.py
- Test with: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`
- If False, try camera index 1, 2, etc: `cap = cv2.VideoCapture(1)`
- Or use a test video file instead of live webcam

---

## 📚 Key Hyperparameters (from project document)

These are all in `src/config.py` but important to know:

```python
IMG_SIZE = (224, 224)           # MobileNetV2 standard
BATCH_SIZE = 32
EPOCHS = 50 (initial) + 10 (fine-tuning)
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.4
TRAIN/VAL/TEST = 70/15/15
AUGMENTATION: Random flip, rotate ±15°, zoom ±10%, brightness ±10%
```

---

## 🎓 Project Structure Alignment

This implementation follows the project document exactly:

- **Chapter 1**: Background, problem, objectives ✅
- **Chapter 2**: Literature review ✅
- **Chapter 3**: Methodology ← This code implements it
- **Chapter 4**: Results ← Will be filled after training
- **Chapter 5**: Conclusion ✅

The code structure mirrors Chapter 3:
- `config.py` → Section 3.4 Software Specifications
- `data_prep.py` → Section 3.1 Data gathering
- `model.py` + `train.py` → Section 3.2 Architecture + Training
- `evaluate.py` → Section 3.6 Evaluation metrics
- `realtime.py` → Real-time system design
- `streamlit_app.py` → User interface

---

## ✨ Next: Run Phase 0

```bash
# Open Anaconda Prompt and run:
cd c:\Users\User\Desktop\PROJECTS\Adeogo\Adeogo
SETUP_CONDA.bat
```

This script will install GPU TensorFlow and all dependencies. Once complete, you can proceed to dataset download.

---

**Questions?** Check the inline code comments in each `.py` file for detailed explanations.
