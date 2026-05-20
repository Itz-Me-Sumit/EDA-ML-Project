import os
from pathlib import Path

# Base Directory path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create Dictionaries if they don't exist
os.makedirs(RAW_DATA_DIR , exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR , exist_ok=True)