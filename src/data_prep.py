r"""
Dataset preparation: Download PlantVillage from Kaggle, filter to solanaceous crops,
and split into train/val/test sets (70/15/15).

BEFORE RUNNING:
1. Download kaggle.json from https://www.kaggle.com/settings/account
2. Place kaggle.json at: C:\Users\User\.kaggle\kaggle.json
3. Then run: python src/data_prep.py
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Set Kaggle API token if available
if os.getenv('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_CONFIG_DIR'] = str(Path.home() / '.kaggle')

from config import (
    RAW_DATA_DIR, PROC_DATA_DIR, MODELS_DIR,
    SOLANACEOUS_CLASSES, SOLANACEOUS_PREFIXES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    KAGGLE_DATASET, CLASS_NAMES_PATH
)

random.seed(RANDOM_SEED)


def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> bool:
    """
    Download PlantVillage dataset from Kaggle using kaggle CLI.

    Args:
        dataset_name: Kaggle dataset slug (e.g., 'vipoooool/new-plant-diseases-dataset')
        output_dir: Directory to download to

    Returns:
        True if successful, False otherwise
    """
    print(f"[*] Downloading dataset: {dataset_name}")
    print(f"[*] Destination: {output_dir}")
    print(f"[*] This may take 5-10 minutes (~1.4 GB)...")

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess

        # Check if KAGGLE_API_TOKEN is set
        token = os.getenv('KAGGLE_API_TOKEN')
        if token:
            print(f"[*] Using KAGGLE_API_TOKEN from environment")

        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir), "--unzip"],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("[✓] Dataset downloaded and extracted successfully")
            return True
        else:
            print(f"[✗] Error downloading dataset: {result.stderr}")
            print(f"[*] stdout: {result.stdout}")
            print(f"\n[!] TROUBLESHOOTING:")
            print(f"    1. Verify token is set: echo %KAGGLE_API_TOKEN%")
            print(f"    2. Verify kaggle.json exists: C:\\.kaggle\\kaggle.json")
            print(f"    3. Try manual download from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
            return False
    except FileNotFoundError:
        print("[✗] ERROR: kaggle CLI not found. Make sure it's installed:")
        print("    pip install kaggle")
        return False
    except Exception as e:
        print(f"[✗] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_source_dirs(raw_root: Path) -> Path:
    """
    Locate the actual train/valid folders inside the downloaded dataset.
    PlantVillage may have nested directories - walk to find class folders.

    Returns:
        Path to the directory containing class subdirectories
    """
    print("[*] Searching for dataset structure...")

    # First, check for augmented dataset folder
    augmented_folder = raw_root / "New Plant Diseases Dataset(Augmented)"
    if augmented_folder.exists():
        # Check if it has train folder
        train_path = augmented_folder / "train"
        if train_path.exists():
            class_dirs = [d for d in os.listdir(train_path) if (Path(train_path) / d).is_dir()]
            if class_dirs:
                print(f"[✓] Found augmented dataset at: {train_path}")
                print(f"[✓] Contains {len(class_dirs)} class folders")
                return train_path

    # Look for 'train' folder anywhere
    for root, dirs, files in os.walk(raw_root):
        if "train" in dirs:
            train_path = Path(root) / "train"
            # Check if this train folder has class subdirectories
            class_dirs = [d for d in os.listdir(train_path) if (Path(train_path) / d).is_dir()]
            if class_dirs:
                print(f"[✓] Found dataset at: {train_path}")
                print(f"[✓] Contains {len(class_dirs)} class folders")
                return train_path

    # If no train folder found, check if raw_root itself contains classes
    dirs = [d for d in os.listdir(raw_root) if (raw_root / d).is_dir()]
    if any(d.startswith(SOLANACEOUS_PREFIXES) for d in dirs):
        print(f"[✓] Found dataset structure in: {raw_root}")
        return raw_root

    raise FileNotFoundError(f"Could not find PlantVillage dataset in {raw_root}")


def get_solanaceous_classes(source_root: Path) -> list[Path]:
    """
    Return list of class directories that match solanaceous crops.

    Args:
        source_root: Root directory containing class subdirectories

    Returns:
        List of Path objects for the 15 target class directories
    """
    class_dirs = []
    all_dirs = os.listdir(source_root)

    print(f"[*] Filtering solanaceous crops from {len(all_dirs)} total classes...")

    for class_name in all_dirs:
        class_path = source_root / class_name
        if not class_path.is_dir():
            continue

        # Check if this class starts with a solanaceous prefix
        if any(class_name.startswith(prefix) for prefix in SOLANACEOUS_PREFIXES):
            class_dirs.append((class_name, class_path))

    # Sort for consistent ordering
    class_dirs.sort(key=lambda x: x[0])

    print(f"[✓] Found {len(class_dirs)} solanaceous classes:")
    for name, _ in class_dirs:
        print(f"    - {name}")

    return [path for _, path in class_dirs]


def split_class_images(class_dir: Path, class_name: str,
                       train_dir: Path, val_dir: Path, test_dir: Path) -> dict:
    """
    Split images from a single class into train/val/test directories.
    Uses stratified random split.

    Returns:
        dict with split counts: {'train': n, 'val': n, 'test': n}
    """
    # Get all images in the class
    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        return {'train': 0, 'val': 0, 'test': 0}

    # Shuffle
    random.shuffle(images)

    # Calculate split indices
    total = len(images)
    train_count = int(total * TRAIN_SPLIT)
    val_count = int(total * VAL_SPLIT)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Create class subdirectories in each split
    train_class_dir = train_dir / class_name
    val_class_dir = val_dir / class_name
    test_class_dir = test_dir / class_name

    for d in [train_class_dir, val_class_dir, test_class_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in train_images:
        src = class_dir / img
        dst = train_class_dir / img
        shutil.copy2(src, dst)

    for img in val_images:
        src = class_dir / img
        dst = val_class_dir / img
        shutil.copy2(src, dst)

    for img in test_images:
        src = class_dir / img
        dst = test_class_dir / img
        shutil.copy2(src, dst)

    return {'train': len(train_images), 'val': len(val_images), 'test': len(test_images)}


def build_processed_dataset(raw_root: Path = RAW_DATA_DIR,
                            proc_root: Path = PROC_DATA_DIR) -> dict:
    """
    Main entry point. Download dataset, filter, and split.

    Returns:
        Summary dict with class counts per split
    """
    # Create processing directories
    train_dir = proc_root / "train"
    val_dir = proc_root / "val"
    test_dir = proc_root / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Check if dataset is already present
    print("\n[STEP 1] Checking for dataset...")

    # Look for the augmented dataset folder
    augmented_folder = raw_root / "New Plant Diseases Dataset(Augmented)"
    if augmented_folder.exists():
        print(f"[✓] Found augmented dataset at: {augmented_folder}")
    elif (raw_root / "train").exists():
        print(f"[✓] Found train folder at: {raw_root / 'train'}")
    else:
        print("\n[STEP 1b] Dataset not found, downloading from Kaggle...")
        if not download_kaggle_dataset(KAGGLE_DATASET, raw_root):
            print("[✗] Failed to download dataset. Check kaggle.json setup.")
            return {}

    # Step 2: Find source directories
    print("\n[STEP 2] Locating dataset structure...")
    try:
        source_root = find_source_dirs(raw_root)
    except FileNotFoundError as e:
        print(f"[✗] {e}")
        return {}

    # Step 3: Filter solanaceous classes
    print("\n[STEP 3] Filtering solanaceous crops...")
    class_dirs = get_solanaceous_classes(source_root)

    if len(class_dirs) < 15:
        print(f"[!] WARNING: Found only {len(class_dirs)} classes, expected 15")

    # Step 4: Split each class into train/val/test
    print("\n[STEP 4] Splitting dataset (70/15/15)...")
    summary = {}

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        counts = split_class_images(class_dir, class_name, train_dir, val_dir, test_dir)
        summary[class_name] = counts

    # Step 5: Save class names mapping
    print("\n[STEP 5] Saving class names mapping...")
    class_names_list = sorted([d.name for d in class_dirs])
    class_names_dict = {i: name for i, name in enumerate(class_names_list)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names_dict, f, indent=2)
    print(f"[✓] Class names saved to: {CLASS_NAMES_PATH}")

    return summary


def verify_dataset(proc_root: Path = PROC_DATA_DIR) -> None:
    """
    Verify dataset structure and print summary statistics.
    """
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70)

    splits = {'train': proc_root / 'train', 'val': proc_root / 'val', 'test': proc_root / 'test'}

    for split_name, split_path in splits.items():
        if not split_path.exists():
            print(f"\n[✗] {split_name} directory not found at {split_path}")
            continue

        class_dirs = [d for d in os.listdir(split_path) if (split_path / d).is_dir()]
        class_dirs.sort()

        print(f"\n{split_name.upper()} SET ({split_path}):")
        print(f"  Classes: {len(class_dirs)}")

        total_images = 0
        for class_dir in class_dirs:
            class_path = split_path / class_dir
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
            print(f"    {class_dir:45s} : {len(images):5d} images")

        print(f"  Total: {total_images} images")

    # Check class names file
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"\n[✓] Class names file exists: {len(class_names)} classes")
    else:
        print(f"\n[✗] Class names file not found: {CLASS_NAMES_PATH}")

    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PLANT DISEASE DETECTION - DATASET PREPARATION")
    print("="*70)
    print(f"Project: Landmark University - B.Sc. Computer Science")
    print(f"Author: Ekunjesu Adeogo (22CD009343)")
    print(f"Target crops: Tomato, Potato, Bell Pepper (15 classes)")
    print("="*70 + "\n")

    try:
        # Build dataset
        summary = build_processed_dataset()

        # Verify
        verify_dataset()

        print("\n[✓] Dataset preparation complete!")
        print("\nNext steps:")
        print("  1. Review the data in data/processed/")
        print("  2. Run: python src/train.py (or use 02_model_training.ipynb)")

    except Exception as e:
        print(f"\n[✗] Error: {e}")
        import traceback
        traceback.print_exc()
