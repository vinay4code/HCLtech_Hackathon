import json
import pandas as pd
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.preprocessing import PreprocessingPipeline
from backend.feature_engineering import FeatureEngineering
from backend.data_loader import DataLoader

class CustomerProfile(BaseModel):
    customer_id: str
    tenure_months: int
    tenure_category: str
    contract: str
    monthly_charges: float
    total_charges: float
    payment_method: str
    paperless_billing: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    subscribed_services: Dict[str, str]
    service_gaps: list[str]
    subscribed_addons_count: int
    has_active_service: bool
    data_quality_warning: bool
    data_quality_notes: str
    
    # Supplementary Simulated Operational Data (labeled explicitly)
    simulated_billing_status: Optional[str] = "Current"
    simulated_outstanding_balance: Optional[float] = 0.0
    simulated_due_date: Optional[str] = "N/A"
    simulated_email_consent: Optional[bool] = True
    simulated_sms_consent: Optional[bool] = False
    simulated_opt_out_all: Optional[bool] = False
    
    # Audit safety notice
    protected_characteristics_excluded: bool = True

class ProfileBuilder:
    """
    Assembles a comprehensive, explainable profile for a telecom customer,
    strictly distinguishing ground-truth dataset attributes from simulated operational fields.
    """
    _enriched_df: Optional[pd.DataFrame] = None

    @classmethod
    def get_enriched_dataframe(cls, reload: bool = False) -> pd.DataFrame:
        if cls._enriched_df is None or reload:
            cleaned_df = PreprocessingPipeline.process_telecom_data(reload=reload)
            cls._enriched_df = FeatureEngineering.enrich_dataframe(cleaned_df)
        return cls._enriched_df.copy()

    @classmethod
    def build_profile(cls, customer_id: str) -> CustomerProfile:
        df = cls.get_enriched_dataframe()
        match = df[df["customerID"] == customer_id]
        if match.empty:
            raise ValueError(f"Customer '{customer_id}' not found.")
        
        row = match.iloc[0]
        
        # Subscribed optional services
        services_dict = {
            "OnlineSecurity": str(row["OnlineSecurity"]),
            "OnlineBackup": str(row["OnlineBackup"]),
            "DeviceProtection": str(row["DeviceProtection"]),
            "TechSupport": str(row["TechSupport"]),
            "StreamingTV": str(row["StreamingTV"]),
            "StreamingMovies": str(row["StreamingMovies"]),
        }

        # Retrieve simulated supplementary data
        sim_billing = cls._get_simulated_billing(customer_id)
        sim_consent = cls._get_simulated_consent(customer_id)

        profile = CustomerProfile(
            customer_id=customer_id,
            tenure_months=int(row["tenure"]),
            tenure_category=str(row["tenure_category"]),
            contract=str(row["Contract"]),
            monthly_charges=float(row["MonthlyCharges"]),
            total_charges=float(row["TotalCharges"]),
            payment_method=str(row["PaymentMethod"]),
            paperless_billing=str(row["PaperlessBilling"]),
            phone_service=str(row["PhoneService"]),
            multiple_lines=str(row["MultipleLines"]),
            internet_service=str(row["InternetService"]),
            subscribed_services=services_dict,
            service_gaps=list(row["service_gaps"]),
            subscribed_addons_count=int(row["subscribed_addons_count"]),
            has_active_service=bool(row["has_active_service"]),
            data_quality_warning=bool(row["data_quality_warning"]),
            data_quality_notes=str(row["data_quality_notes"]),
            simulated_billing_status=sim_billing.get("payment_status", "Current"),
            simulated_outstanding_balance=float(sim_billing.get("outstanding_balance", 0.0)),
            simulated_due_date=str(sim_billing.get("due_date", "2026-11-28")),
            simulated_email_consent=bool(sim_consent.get("email_marketing_consent", True)),
            simulated_sms_consent=bool(sim_consent.get("sms_consent", False)),
            simulated_opt_out_all=bool(sim_consent.get("opt_out_all", False)),
            protected_characteristics_excluded=True
        )
        return profile

    @classmethod
    def _get_simulated_billing(cls, customer_id: str) -> Dict[str, Any]:
        try:
            b_df = DataLoader.load_simulated_billing()
            m = b_df[b_df["customerID"] == customer_id]
            if not m.empty:
                return m.iloc[0].to_dict()
        except Exception:
            pass
        return {}

    @classmethod
    def _get_simulated_consent(cls, customer_id: str) -> Dict[str, Any]:
        try:
            c_df = DataLoader.load_simulated_consent()
            m = c_df[c_df["customerID"] == customer_id]
            if not m.empty:
                return m.iloc[0].to_dict()
        except Exception:
            pass
        return {}
