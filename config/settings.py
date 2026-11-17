import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OFFERS_DOCS_DIR = DATA_DIR / "offers"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
DB_PATH = DATA_DIR / "assistant.db"

TELECOM_CSV_PATH = DATA_DIR / "Telecom_dataset.csv"
SIMULATED_BILLING_PATH = DATA_DIR / "simulated_billing.csv"
SIMULATED_CONSENT_PATH = DATA_DIR / "simulated_consent.csv"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OFFERS_DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configurations
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "MOCK") # "GEMINI", "OPENAI", or "MOCK"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-1.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Embedding Configurations
LOCAL_HF_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "FAST_HASH") # "FAST_HASH", "LOCAL_HF", "GEMINI"


# App Branding
APP_NAME = "Customer Loyalty & Engagement Assistant"
APP_SUBTITLE = "HCLTech Future Ready AI Engineer Hackathon"
APP_VERSION = "2.0.0"
COMPANY_NAME = "NovaTel Telecom Solutions"
SUPPORT_EMAIL = "support@novatel.example.com"
SUPPORT_PHONE = "1-800-555-NOVA"
OPT_OUT_LINK = "https://novatel.example.com/preferences/manage"
