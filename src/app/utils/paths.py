from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

SRC_DIR = ROOT_DIR / "src"

CONFIGS_DIR = ROOT_DIR / "src" / "configs"
DATA_DIR = ROOT_DIR / "src" / "data"

def config_file(name: str = "config.json") -> Path:
	return CONFIGS_DIR / name

def dataset_file(name: str) -> Path:
	return DATA_DIR / name
