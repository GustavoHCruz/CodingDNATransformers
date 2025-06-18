from pathlib import Path

current_dir = Path(__file__).resolve()

project_root = current_dir.parents[3]

storage_dir = project_root / "storage"