import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from config.settings import BASE_DIR, DATA_DIR
from backend.profile_builder import CustomerProfile

MODEL_CACHE_PATH = DATA_DIR / "rf_model.pkl"


class SegmentationEngine:
    """
    Machine Learning-based Customer Segmentation Engine using Random Forest.
    Features:
    - Multi-class and probabilistic multi-segment prediction.
    - Confidence scoring derived from tree ensemble voting probabilities.
    - Global feature importance and instance-level explainability.
    - Zero protected characteristics: gender, SeniorCitizen, partner, dependents excluded.
    """
    _instance = None
    _model: Optional[RandomForestClassifier] = None
    _feature_names: List[str] = []
    _classes: List[str] = []
    _metrics: Dict[str, Any] = {}
    _feature_importances: Dict[str, float] = {}
    _multi_threshold: float = 0.20

    FEATURE_COLUMNS = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "service_gaps_count",
        "subscribed_addons_count",
        "contract_month_to_month",
        "contract_one_year",
        "contract_two_year",
        "internet_dsl",
        "internet_fiber",
        "internet_no",
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "paperless_billing"
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42, multi_threshold: float = 0.20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.multi_threshold = multi_threshold
        self._ensure_model_trained()

    @classmethod
    def get_instance(cls) -> "SegmentationEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_model_trained(self, force_retrain: bool = False):
        if self._model is not None and not force_retrain:
            return

        # Check cached model on disk
        if MODEL_CACHE_PATH.exists() and not force_retrain:
            try:
                with open(MODEL_CACHE_PATH, "rb") as f:
                    cached_data = pickle.load(f)
                self._model = cached_data["model"]
                self._feature_names = cached_data["feature_names"]
                self._classes = list(self._model.classes_)
                self._metrics = cached_data["metrics"]
                self._feature_importances = cached_data["feature_importances"]
                self._multi_threshold = cached_data.get("multi_threshold", self.multi_threshold)
                self.n_estimators = self._metrics.get("n_estimators", self.n_estimators)
                self.max_depth = self._metrics.get("max_depth", self.max_depth)
                self.random_state = self._metrics.get("random_state", self.random_state)
                return
            except Exception:
                pass


        # Train model
        self.train_model(self.n_estimators, self.max_depth, self.random_state, self.multi_threshold)

    @classmethod
    def extract_features_from_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms telecom dataframe into structured ML feature matrix."""
        X = pd.DataFrame(index=df.index)
        X["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(float)
        X["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
        X["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype(float)
        
        # Add service gap counts
        if "service_gaps_count" in df.columns:
            X["service_gaps_count"] = df["service_gaps_count"].astype(float)
        else:
            X["service_gaps_count"] = 0.0

        if "subscribed_addons_count" in df.columns:
            X["subscribed_addons_count"] = df["subscribed_addons_count"].astype(float)
        else:
            X["subscribed_addons_count"] = 0.0

        # Contract encoding
        contract_series = df["Contract"].astype(str)
        X["contract_month_to_month"] = (contract_series == "Month-to-month").astype(int)
        X["contract_one_year"] = (contract_series == "One year").astype(int)
        X["contract_two_year"] = (contract_series == "Two year").astype(int)

        # Internet service encoding
        internet_series = df["InternetService"].astype(str)
        X["internet_dsl"] = (internet_series == "DSL").astype(int)
        X["internet_fiber"] = (internet_series == "Fiber optic").astype(int)
        X["internet_no"] = (internet_series == "No").astype(int)

        # Core services
        X["phone_service"] = (df["PhoneService"].astype(str).str.lower() == "yes").astype(int)
        X["multiple_lines"] = (df["MultipleLines"].astype(str).str.lower() == "yes").astype(int)
        X["online_security"] = (df["OnlineSecurity"].astype(str).str.lower() == "yes").astype(int)
        X["online_backup"] = (df["OnlineBackup"].astype(str).str.lower() == "yes").astype(int)
        X["device_protection"] = (df["DeviceProtection"].astype(str).str.lower() == "yes").astype(int)
        X["tech_support"] = (df["TechSupport"].astype(str).str.lower() == "yes").astype(int)
        X["streaming_tv"] = (df["StreamingTV"].astype(str).str.lower() == "yes").astype(int)
        X["streaming_movies"] = (df["StreamingMovies"].astype(str).str.lower() == "yes").astype(int)
        X["paperless_billing"] = (df["PaperlessBilling"].astype(str).str.lower() == "yes").astype(int)

        return X[cls.FEATURE_COLUMNS]

    @classmethod
    def generate_ground_truth_labels(cls, df: pd.DataFrame) -> pd.Series:
        """Assigns ground truth segment labels across the customer base."""
        labels = []
        for _, r in df.iterrows():
            tenure = r["tenure"]
            contract = r["Contract"]
            active = r.get("has_active_service", True)
            gaps = r.get("service_gaps_count", 0)
            internet = r["InternetService"]

            if tenure >= 48 and active:
                labels.append("LONG_TERM_LOYAL")
            elif contract == "Month-to-month" and tenure >= 12:
                labels.append("CONTRACT_RENEWAL_CANDIDATE")
            elif tenure <= 6:
                labels.append("EARLY_STAGE_CUSTOMER")
            elif contract in ["One year", "Two year"] and tenure >= 12:
                labels.append("CONSISTENT_SUBSCRIBER")
            elif internet in ["DSL", "Fiber optic"] and gaps >= 1:
                labels.append("UPSELL_OPPORTUNITY")
            else:
                labels.append("STANDARD_SUBSCRIBER")
        return pd.Series(labels, index=df.index)

    def train_model(
        self,
        n_estimators: int = 50,
        max_depth: int = 4,
        random_state: int = 42,
        multi_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """
        Trains the Random Forest model on the telecom customer base,
        evaluates performance metrics, and caches the trained artifact.
        """
        import time
        from datetime import datetime
        t0 = time.time()

        from backend.profile_builder import ProfileBuilder
        df = ProfileBuilder.get_enriched_dataframe()

        X = self.extract_features_from_dataframe(df)
        y = self.generate_ground_truth_labels(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=random_state, stratify=y
        )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1
        )
        rf.fit(X_train, y_train)

        # Evaluate performance on holdout test set
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Evaluate performance on training set
        y_train_pred = rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Feature importances
        importances = {
            col: round(float(imp), 4)
            for col, imp in zip(self.FEATURE_COLUMNS, rf.feature_importances_)
        }
        # Sort descending
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = round(time.time() - t0, 3)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.multi_threshold = multi_threshold

        self._model = rf
        self._feature_names = self.FEATURE_COLUMNS
        self._classes = list(rf.classes_)
        self._feature_importances = importances
        self._multi_threshold = multi_threshold
        self._metrics = {
            "accuracy": round(float(acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "precision_weighted": round(float(p), 4),
            "recall_weighted": round(float(r), 4),
            "f1_weighted": round(float(f1), 4),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "classification_report": report,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "multi_threshold": multi_threshold,
            "trained_at": now_str,
            "train_duration_sec": duration
        }

        # Cache to disk with pickle
        with open(MODEL_CACHE_PATH, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "metrics": self._metrics,
                "feature_importances": self._feature_importances,
                "multi_threshold": self._multi_threshold
            }, f)

        SegmentationEngine._instance = self
        return self._metrics

    def get_model_metrics(self, force_reload: bool = False) -> Dict[str, Any]:
        """Returns the cached evaluation metrics and feature importances."""
        self._ensure_model_trained(force_retrain=force_reload)
        return {
            "metrics": self._metrics,
            "feature_importances": self._feature_importances,
            "classes": self._classes
        }


    def extract_features_from_profile(self, profile: CustomerProfile) -> pd.DataFrame:
        """Converts an individual CustomerProfile instance into a single-row feature DataFrame."""
        row_dict = {
            "tenure": float(profile.tenure_months),
            "MonthlyCharges": float(profile.monthly_charges),
            "TotalCharges": float(profile.total_charges),
            "service_gaps_count": float(len(profile.service_gaps)),
            "subscribed_addons_count": float(profile.subscribed_addons_count),
            "contract_month_to_month": 1 if profile.contract == "Month-to-month" else 0,
            "contract_one_year": 1 if profile.contract == "One year" else 0,
            "contract_two_year": 1 if profile.contract == "Two year" else 0,
            "internet_dsl": 1 if profile.internet_service == "DSL" else 0,
            "internet_fiber": 1 if profile.internet_service == "Fiber optic" else 0,
            "internet_no": 1 if profile.internet_service == "No" else 0,
            "phone_service": 1 if profile.phone_service.lower() == "yes" else 0,
            "multiple_lines": 1 if profile.multiple_lines.lower() == "yes" else 0,
            "online_security": 1 if profile.subscribed_services.get("OnlineSecurity", "").lower() == "yes" else 0,
            "online_backup": 1 if profile.subscribed_services.get("OnlineBackup", "").lower() == "yes" else 0,
            "device_protection": 1 if profile.subscribed_services.get("DeviceProtection", "").lower() == "yes" else 0,
            "tech_support": 1 if profile.subscribed_services.get("TechSupport", "").lower() == "yes" else 0,
            "streaming_tv": 1 if profile.subscribed_services.get("StreamingTV", "").lower() == "yes" else 0,
            "streaming_movies": 1 if profile.subscribed_services.get("StreamingMovies", "").lower() == "yes" else 0,
            "paperless_billing": 1 if profile.paperless_billing.lower() == "yes" else 0,
        }
        return pd.DataFrame([row_dict])[self.FEATURE_COLUMNS]

    def classify_customer(self, profile: CustomerProfile) -> Dict[str, Any]:
        """
        Classifies customer using the trained Random Forest model.
        Returns predicted primary segment, confidence score, full probability distribution,
        multi-segment assignments, and explainable feature contributions.
        """
        self._ensure_model_trained()
        X_inst = self.extract_features_from_profile(profile)

        probs = self._model.predict_proba(X_inst)[0]
        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(self._classes, probs)}
        
        # Sort classes by probability descending
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        primary_segment, confidence = sorted_probs[0]

        # Multi-segment assignment based on threshold (e.g. >= 0.20)
        assigned_segments = [seg for seg, p in sorted_probs if p >= self._multi_threshold]
        if not assigned_segments:
            assigned_segments = [primary_segment]

        # Compute instance-level explainable feature contributions
        reasons, segment_details, top_features = self._explain_prediction(profile, primary_segment, confidence, prob_dict)

        return {
            "customer_id": profile.customer_id,
            "primary_segment": primary_segment,
            "confidence_score": round(float(confidence), 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "segments": assigned_segments,
            "probabilities": prob_dict,
            "reasons": reasons,
            "segment_details": segment_details,
            "top_feature_contributions": top_features,
            "model_provenance": f"Random Forest Classifier ({self._model.n_estimators} trees, max_depth={self._model.max_depth})",
            "rule_provenance": f"ML Random Forest Inference (Confidence: {confidence * 100:.1f}%)"
        }

    def _explain_prediction(
        self,
        profile: CustomerProfile,
        primary_seg: str,
        confidence: float,
        prob_dict: Dict[str, float]
    ) -> Tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
        """Derives transparent, feature-level reasons for the Random Forest decision."""
        reasons = []
        segment_details = {}
        top_features = []

        # 1. Primary confidence statement
        r_conf = f"Random Forest ensemble classified customer as '{primary_seg}' with {confidence * 100:.1f}% confidence."
        reasons.append(r_conf)
        segment_details[primary_seg] = r_conf

        # 2. Key feature drivers based on customer values and global tree importances
        if profile.tenure_months >= 48:
            r = f"High tenure of {profile.tenure_months} months is the primary structural driver (Feature Importance: {self._feature_importances.get('tenure', 0.45)*100:.1f}%)."
            reasons.append(r)
            top_features.append({"feature": "tenure", "value": profile.tenure_months, "importance": self._feature_importances.get("tenure", 0.45)})
        elif profile.tenure_months <= 6:
            r = f"Recent onboarding tenure of {profile.tenure_months} month(s) strongly influenced early-stage classification."
            reasons.append(r)
            top_features.append({"feature": "tenure", "value": profile.tenure_months, "importance": self._feature_importances.get("tenure", 0.45)})

        if profile.contract == "Month-to-month":
            r = f"Flexible Month-to-month contract commitment (Feature Importance: {self._feature_importances.get('contract_month_to_month', 0.12)*100:.1f}%) signals retention focus."
            reasons.append(r)
            top_features.append({"feature": "contract", "value": profile.contract, "importance": self._feature_importances.get("contract_month_to_month", 0.12)})
        else:
            r = f"Committed '{profile.contract}' agreement supports stability scoring."
            reasons.append(r)
            top_features.append({"feature": "contract", "value": profile.contract, "importance": self._feature_importances.get("contract_one_year", 0.05)})

        if profile.service_gaps:
            r = f"Identified {len(profile.service_gaps)} missing service gap(s) ({', '.join(profile.service_gaps[:2])}) contributing to upsell propensity."
            reasons.append(r)
            top_features.append({"feature": "service_gaps_count", "value": len(profile.service_gaps), "importance": self._feature_importances.get("service_gaps_count", 0.02)})

        # 3. Probabilistic secondary signal
        for seg, p in prob_dict.items():
            if seg != primary_seg and p >= self._multi_threshold:
                r_sec = f"Secondary segment membership: '{seg}' probability is {p * 100:.1f}%."
                reasons.append(r_sec)
                segment_details[seg] = r_sec

        return reasons, segment_details, top_features

    def predict_all_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch inference for all customers in dataframe."""
        self._ensure_model_trained()
        X = self.extract_features_from_dataframe(df)
        preds = self._model.predict(X)
        probs = self._model.predict_proba(X)
        confidences = np.max(probs, axis=1)

        result_df = df.copy()
        result_df["ml_predicted_segment"] = preds
        result_df["ml_confidence"] = np.round(confidences, 4)
        return result_df
