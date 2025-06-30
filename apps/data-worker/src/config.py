from pathlib import Path

current_dir = Path(__file__).resolve()

apps_root = current_dir.parents[2]
project_root = current_dir.parents[3]

shared_dir = apps_root / "shared"
storage_dir = project_root / "storage"