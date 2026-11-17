import pandas as pd
from typing import List, Dict, Any

OPTIONAL_INTERNET_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

class FeatureEngineering:
    """
    Computes explainable telecom engagement features and service gap metrics.
    """

    @staticmethod
    def identify_service_gaps(row: pd.Series) -> List[str]:
        """Returns the list of optional services not subscribed by the customer."""
        gaps = []
        has_internet = str(row.get("InternetService", "")).strip().lower() in ["dsl", "fiber optic"]
        
        if has_internet:
            for s in OPTIONAL_INTERNET_SERVICES:
                val = str(row.get(s, "")).strip().lower()
                if val != "yes":
                    gaps.append(s)
        
        has_phone = str(row.get("PhoneService", "")).strip().lower() == "yes"
        if has_phone:
            ml_val = str(row.get("MultipleLines", "")).strip().lower()
            if ml_val != "yes":
                gaps.append("MultipleLines")
                
        return gaps

    @staticmethod
    def count_subscribed_addons(row: pd.Series) -> int:
        """Counts how many optional internet addons the customer subscribes to."""
        count = 0
        for s in OPTIONAL_INTERNET_SERVICES:
            if str(row.get(s, "")).strip().lower() == "yes":
                count += 1
        return count

    @staticmethod
    def get_tenure_category(tenure: int) -> str:
        if tenure <= 6:
            return "0-6m (New)"
        elif tenure <= 12:
            return "7-12m (Early)"
        elif tenure <= 24:
            return "13-24m (Established)"
        elif tenure <= 48:
            return "25-48m (Mature)"
        else:
            return "49-72m (Loyal)"

    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Adds computed features to the dataframe."""
        df = df.copy()
        df["service_gaps"] = df.apply(cls.identify_service_gaps, axis=1)
        df["service_gaps_count"] = df["service_gaps"].apply(len)
        df["subscribed_addons_count"] = df.apply(cls.count_subscribed_addons, axis=1)
        df["tenure_category"] = df["tenure"].apply(cls.get_tenure_category)
        
        df["has_active_service"] = (
            (df["PhoneService"].str.lower() == "yes") |
            (df["InternetService"].str.lower().isin(["dsl", "fiber optic"]))
        )
        return df
