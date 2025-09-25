from pathlib import Path
import os

# Base dirs resolved relative to this file
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models" / "artifacts"
REPORTS_DIR = ROOT_DIR / "models" / "reports"

# Env flags
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "false").lower() == "true"

# Files
APPLICANTS_PATH = RAW_DIR / "applicants.json"
VAGAS_PATH = RAW_DIR / "vagas.json"
PROSPECTS_PATH = RAW_DIR / "prospects.json"

# Training params (could be moved to params.yaml if desired)
RANDOM_STATE = 42
N_JOBS = -1
