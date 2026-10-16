"""统一项目路径，避免依赖启动时的工作目录。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "features"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"
