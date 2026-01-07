# src/my_rag/config.py
import yaml
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../configs")

def load_yaml(filename: str) -> dict:
    path = os.path.join(CONFIG_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

models_config = load_yaml("models.yml")
hyperparams = load_yaml("hyperparameters.yml")
vector_db_config = load_yaml("vector_db.yml")  # ← NEW

# Existing
LLM_MODELS = models_config["llm_models"]
DEFAULT_LLM = models_config["default_llm"]
EMBEDDING_MODELS = models_config["embedding_models"]
DEFAULT_EMBEDDING = models_config["default_embedding"]

CHUNK_SIZE = hyperparams["chunking"]["chunk_size"]
CHUNK_OVERLAP = hyperparams["chunking"]["chunk_overlap"]

# New: Vector DB
VECTOR_DB_DEFAULT = vector_db_config["default"]
VECTOR_DB_CONFIG = vector_db_config