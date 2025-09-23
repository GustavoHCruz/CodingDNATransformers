from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent

STORAGE_PATH = project_root / "storage"

print(STORAGE_PATH)