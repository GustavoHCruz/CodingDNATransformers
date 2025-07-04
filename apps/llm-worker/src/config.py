from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()

APPS_ROOT = CURRENT_DIR.parents[2]
PROJECT_ROOT = CURRENT_DIR.parents[3]

SHARED_DIR = APPS_ROOT / "shared"
STORAGE_DIR = PROJECT_ROOT / "storage"