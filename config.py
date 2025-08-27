import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT.parent / "vbase-data"

# Allow using env variable path
DATA_DIR = Path(os.getenv("VBASE_DATA_DIR", DATA_DIR)).resolve()
