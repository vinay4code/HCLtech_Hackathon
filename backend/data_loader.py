import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from config.settings import TELECOM_CSV_PATH, SIMULATED_BILLING_PATH, SIMULATED_CONSENT_PATH

class DataLoader:
    """
    Loads raw telecom datasets and supplementary simulated datasets.
    Preserves separation between original telecom records and simulated data.
    """
    _cached_telecom_df: Optional[pd.DataFrame] = None
    _cached_billing_df: Optional[pd.DataFrame] = None
    _cached_consent_df: Optional[pd.DataFrame] = None

    @classmethod
    def load_raw_telecom_data(cls, reload: bool = False) -> pd.DataFrame:
        if cls._cached_telecom_df is None or reload:
            if not TELECOM_CSV_PATH.exists():
                raise FileNotFoundError(f"Telecom dataset not found at {TELECOM_CSV_PATH}")
            cls._cached_telecom_df = pd.read_csv(TELECOM_CSV_PATH)
        return cls._cached_telecom_df.copy()

    @classmethod
    def load_simulated_billing(cls, reload: bool = False) -> pd.DataFrame:
        if cls._cached_billing_df is None or reload:
            if not SIMULATED_BILLING_PATH.exists():
                raise FileNotFoundError(f"Simulated billing data not found at {SIMULATED_BILLING_PATH}")
            cls._cached_billing_df = pd.read_csv(SIMULATED_BILLING_PATH)
        return cls._cached_billing_df.copy()

    @classmethod
    def load_simulated_consent(cls, reload: bool = False) -> pd.DataFrame:
        if cls._cached_consent_df is None or reload:
            if not SIMULATED_CONSENT_PATH.exists():
                raise FileNotFoundError(f"Simulated consent data not found at {SIMULATED_CONSENT_PATH}")
            cls._cached_consent_df = pd.read_csv(SIMULATED_CONSENT_PATH)
        return cls._cached_consent_df.copy()
