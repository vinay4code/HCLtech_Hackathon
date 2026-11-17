import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from backend.data_loader import DataLoader

class PreprocessingPipeline:
    """
    Cleans, validates, and prepares telecom data for feature engineering and modeling.
    Guarantees deterministic data type casting and tracks data quality annotations.
    """
    _cached_cleaned_df: pd.DataFrame = None

    @classmethod
    def process_telecom_data(cls, reload: bool = False) -> pd.DataFrame:
        if cls._cached_cleaned_df is not None and not reload:
            return cls._cached_cleaned_df.copy()

        df = DataLoader.load_raw_telecom_data(reload=reload)
        
        # 1. Handle TotalCharges missing/whitespace values
        # In IBM Telco data, customers with tenure == 0 have empty spaces in TotalCharges
        whitespace_mask = df["TotalCharges"].astype(str).str.strip() == ""
        
        # Create data quality warning column
        df["data_quality_warning"] = False
        df["data_quality_notes"] = ""
        
        if whitespace_mask.any():
            df.loc[whitespace_mask, "data_quality_warning"] = True
            df.loc[whitespace_mask, "data_quality_notes"] = "New account (tenure=0): TotalCharges empty, imputed to 0.00"
            df.loc[whitespace_mask, "TotalCharges"] = "0.0"

        # Safe float conversion
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

        # 2. String trimming for categorical columns
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            if col != "data_quality_notes":
                df[col] = df[col].astype(str).str.strip()

        cls._cached_cleaned_df = df
        return df.copy()
