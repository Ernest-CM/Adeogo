"""
Configuration file for Plant Disease Detection System
All hyperparameters and paths defined here
"""
import os
from pathlib import Path

# ===== PROJECT PATHS =====
BASE_DIR = Path(r"c:\Users\User\Desktop\PROJECTS\Adeogo\Adeogo")
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROC_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"

# Create dirs if they don't exist
for d in [RAW_DATA_DIR, PROC_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== MODEL HYPERPARAMETERS (from project document Chapter 4, Table 4.2) =====
IMG_SIZE = (224, 224)           # MobileNetV2 standard input
NUM_CHANNELS = 3                # RGB
BATCH_SIZE = 32                 # Training batch size
EPOCHS = 50                     # Training epochs
LEARNING_RATE = 0.0001          # Adam optimizer learning rate
DROPOUT_RATE = 0.4              # Dropout in fully connected layers
RANDOM_SEED = 42                # For reproducibility

# ===== DATA SPLIT (from project document Section 3.1) =====
TRAIN_SPLIT = 0.70              # 70% training
VAL_SPLIT = 0.15                # 15% validation
TEST_SPLIT = 0.15               # 15% testing

# ===== SOLANACEOUS CROPS (15 classes total, filtered from PlantVillage) =====
# Format: Crop___Disease or Crop___healthy
SOLANACEOUS_CLASSES = [
    # Tomato (10 classes)
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    # Potato (3 classes)
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    # Bell Pepper (2 classes)
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
]

NUM_CLASSES = len(SOLANACEOUS_CLASSES)  # 15

# Dataset prefixes to filter from PlantVillage (raw download)
SOLANACEOUS_PREFIXES = ("Tomato", "Potato", "Pepper")

# ===== MODEL SAVE PATHS =====
MODEL_SAVE_PATH = MODELS_DIR / "plant_disease_mobilenetv2.keras"
TFLITE_SAVE_PATH = MODELS_DIR / "plant_disease.tflite"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
TRAINING_LOG_PATH = MODELS_DIR / "training_log.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"

# ===== AUGMENTATION PARAMETERS (from project document Section 3.3.1) =====
ROTATION_FACTOR = 0.15          # ±15 degrees
ZOOM_FACTOR = 0.10              # ±10%
BRIGHTNESS_FACTOR = 0.10        # ±10%
CONTRAST_FACTOR = 0.10          # ±10%

# ===== TRAINING CALLBACKS =====
EARLY_STOPPING_PATIENCE = 10    # Stop if val_loss doesn't improve for 10 epochs
REDUCE_LR_PATIENCE = 5          # Reduce LR if val_loss doesn't improve for 5 epochs
REDUCE_LR_FACTOR = 0.5          # Multiply LR by 0.5 when triggered
MIN_LEARNING_RATE = 1e-7        # Minimum LR floor

# ===== TRAINING PHASES =====
INITIAL_TRAINING_EPOCHS = 50    # First pass with frozen base
FINE_TUNING_EPOCHS = 10         # Second pass with unfrozen top layers
FINE_TUNING_LEARNING_RATE = 1e-5  # Lower LR for fine-tuning

# ===== INFERENCE =====
CONFIDENCE_THRESHOLD = 0.60     # Default threshold for Streamlit app
TOP_K_PREDICTIONS = 3           # Return top-3 predictions

# ===== KAGGLE DATASET =====
KAGGLE_DATASET = "vipoooool/new-plant-diseases-dataset"
KAGGLE_DATASET_SIZE_MB = 1400   # Approximate download size

# ===== ENVIRONMENT =====
try:
    GOOGLE_COLAB = 'google.colab' in str(__import__('sys').modules)
except:
    GOOGLE_COLAB = False

# ===== DISEASE INFO DATABASE =====
# Hardcoded disease descriptions for UI display
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "description": "Bacterial leaf spot on tomato",
        "symptoms": ["Small, dark, oily-looking spots on leaves", "Spots may have a yellow halo"],
        "treatment": ["Remove affected leaves", "Improve airflow", "Apply copper-based fungicides"]
    },
    "Tomato___Early_blight": {
        "description": "Early blight caused by Alternaria alternata fungus",
        "symptoms": ["Brown circular spots with concentric rings on lower leaves", "Spots enlarge gradually"],
        "treatment": ["Remove lower leaves", "Apply fungicide (mancozeb)", "Ensure good ventilation"]
    },
    "Tomato___Late_blight": {
        "description": "Late blight caused by Phytophthora infestans (fungal-like pathogen)",
        "symptoms": ["Water-soaked spots on leaves and stems", "White fungal growth on undersides"],
        "treatment": ["Apply fungicides (chlorothalonil)", "Remove affected parts", "Improve airflow"]
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf mold caused by Passalora fulva fungus",
        "symptoms": ["Pale green spots on leaves", "Yellow halos", "Gray-brown mold on undersides"],
        "treatment": ["Reduce humidity", "Improve ventilation", "Apply fungicides (sulfur-based)"]
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot caused by Septoria lycopersici fungus",
        "symptoms": ["Small circular spots with dark borders", "Gray centers", "Water-soaked appearance"],
        "treatment": ["Remove infected leaves", "Apply fungicides (copper-based)", "Improve airflow"]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-spotted spider mite infestation (pest damage, not disease)",
        "symptoms": ["Fine webbing on leaves", "Stippled/mottled yellow patches", "Distorted growth"],
        "treatment": ["Increase humidity", "Spray with water", "Apply miticide or neem oil"]
    },
    "Tomato___Target_Spot": {
        "description": "Target spot caused by Corynespora cassiicola fungus",
        "symptoms": ["Circular spots with concentric rings (target pattern)", "Affects leaves and stems"],
        "treatment": ["Remove affected leaves", "Apply fungicides", "Maintain good ventilation"]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) spread by whiteflies",
        "symptoms": ["Yellowing and curling of leaves", "Stunted growth", "No fruit development"],
        "treatment": ["Control whiteflies", "Remove infected plants", "Use resistant varieties"]
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato Mosaic Virus (ToMV) spread by contact or tools",
        "symptoms": ["Mosaic pattern of light and dark patches", "Mottled appearance on leaves"],
        "treatment": ["Remove infected plants", "Disinfect tools", "Wash hands", "Use resistant varieties"]
    },
    "Tomato___healthy": {
        "description": "Healthy tomato plant",
        "symptoms": ["No visible disease symptoms", "Green, unblemished leaves"],
        "treatment": ["Water regularly", "Provide sunlight", "Monitor for pests"]
    },
    "Potato___Early_blight": {
        "description": "Early blight on potato caused by Alternaria solani fungus",
        "symptoms": ["Brown circular spots on lower leaves", "Concentric rings", "Spots coalesce"],
        "treatment": ["Remove lower leaves", "Apply fungicides (mancozeb)", "Ensure good airflow"]
    },
    "Potato___Late_blight": {
        "description": "Late blight on potato caused by Phytophthora infestans",
        "symptoms": ["Water-soaked lesions on leaves and tubers", "White fungal growth on undersides"],
        "treatment": ["Apply fungicides", "Remove affected plants", "Store tubers in cool/dry conditions"]
    },
    "Potato___healthy": {
        "description": "Healthy potato plant",
        "symptoms": ["No visible disease symptoms", "Green foliage", "Normal growth"],
        "treatment": ["Monitor for pests/diseases", "Water at soil level", "Maintain proper nutrition"]
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot on bell pepper caused by Xanthomonas species",
        "symptoms": ["Small dark oily spots on leaves and fruit", "Spots enlarge with yellow halos"],
        "treatment": ["Remove affected parts", "Apply copper fungicides", "Improve air circulation"]
    },
    "Pepper,_bell___healthy": {
        "description": "Healthy bell pepper plant",
        "symptoms": ["No visible disease symptoms", "Healthy foliage and fruit development"],
        "treatment": ["Consistent watering", "Proper nutrition", "Monitor plant health"]
    },
}

# ===== DEBUG / LOGGING =====
DEBUG = False                   # Set to True for verbose output
LOG_BATCH_PREDICTIONS = False   # Log individual predictions during training
