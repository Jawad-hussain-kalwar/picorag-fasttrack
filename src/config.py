import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {"'", '"'} and value[-1] == value[0]:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(ENV_PATH)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


# Log verbosity presets (set with LOG_VERBOSITY in .env):
# - "silent": only errors are shown.
# - "quiet": warnings and errors are shown (default).
# - "normal": info, warnings, and errors are shown.
# - "debug": debug + all higher levels are shown.
LOG_VERBOSITY = os.getenv("LOG_VERBOSITY", "quiet").strip().lower()
VERBOSITY_TO_LOG_LEVEL = {
    "silent": "ERROR",
    "quiet": "WARNING",
    "normal": "INFO",
    "debug": "DEBUG",
}
LOG_LEVEL = VERBOSITY_TO_LOG_LEVEL.get(LOG_VERBOSITY, "WARNING")
LOG_COLOR = _env_bool("LOG_COLOR", default=True)

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_data"
COLLECTION_NAME = "documents"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it")
OPENROUTER_TIMEOUT_SECONDS = 60.0
OPENROUTER_MAX_TOKENS = 300
OPENROUTER_TEMPERATURE = 0.2
OPENROUTER_RATE_LIMIT = int(os.getenv("OPENROUTER_RATE_LIMIT", "600"))

TOP_K = 3

# --- MIRAGE experiment settings ---
MIRAGE_DIR = PROJECT_ROOT / "MIRAGE" / "mirage"
MIRAGE_COLLECTION_NAME = "mirage_doc_pool"
RUNS_DIR = PROJECT_ROOT / "runs"
E1_K_VALUES = [3, 5, 10]
E1_PARTIAL_N = 100

# --- E2 experiment settings ---
E2_K_VALUES = [3, 5, 10]
E2_EVAL_N = 100
E2_INDEX_N = 500
E2_RERANK_TOP_N = 25
E2_RRF_K = 30
VOYAGE_RERANK_MODEL = "rerank-2.5-lite"
OPENROUTER_EMBED_MODEL = "qwen/qwen3-embedding-4b"
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# --- E3 experiment settings ---
# --- Judge settings ---
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "z-ai/glm-4.7-flash")
JUDGE_RATE_LIMIT = int(os.getenv("JUDGE_RATE_LIMIT", "600"))

# --- E3 experiment settings ---
E3_THRESHOLDS = [0.3, 0.5, 0.7]
E3_ABSTAIN_MESSAGE = "Not enough evidence in knowledge base"
E3_LOCAL_BEST_CONFIG = "1_vector_minilm"  # placeholder, update after E2
E3_LOCAL_BEST_K = 3                        # placeholder
