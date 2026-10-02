"""Statistical feature drift detection and distribution shift monitoring."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from production_rag_evaluation.features.models import (
    FeatureDriftReport,
    FeatureGroup,
    FeatureRecord,
)
from production_rag_evaluation.features.pipeline import StreamingFeaturePipeline
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.schemas import AlertSeverity, TelemetryAlert


class FeatureDriftDetector:
    """Detects statistical feature distribution shifts and Population Stability Index (PSI) drift."""

    def __init__(
        self,
        store: ParquetFeatureStore,
        pipeline: Optional[StreamingFeaturePipeline] = None,
    ) -> None:
        self.store = store
        self.pipeline = pipeline

    @staticmethod
    def calculate_psi(
        baseline: np.ndarray,
        target: np.ndarray,
        num_buckets: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Calculate Population Stability Index (PSI) between two numeric distributions."""
        if len(baseline) == 0 or len(target) == 0:
            return 0.0

        b_clean = baseline[~np.isnan(baseline)]
        t_clean = target[~np.isnan(target)]

        if len(b_clean) < 2 or len(t_clean) < 2:
            return 0.0

        # If distributions are identical or constant
        if np.all(b_clean == b_clean[0]) and np.all(t_clean == t_clean[0]):
            return 0.0 if b_clean[0] == t_clean[0] else 1.0

        # Create bucket boundaries based on baseline percentiles
        quantiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(b_clean, quantiles)
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 2:
            # Fall back to min/max range
            min_val = min(float(np.min(b_clean)), float(np.min(t_clean)))
            max_val = max(float(np.max(b_clean)), float(np.max(t_clean)))
            if min_val == max_val:
                return 0.0
            breakpoints = np.linspace(min_val, max_val, num_buckets + 1)

        # Extend outer edges to cover target bounds
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Calculate counts per bucket
        b_counts, _ = np.histogram(b_clean, bins=breakpoints)
        t_counts, _ = np.histogram(t_clean, bins=breakpoints)

        # Calculate fraction in each bucket
        b_pct = b_counts / float(len(b_clean))
        t_pct = t_counts / float(len(t_clean))

        # Apply epsilon smoothing to avoid log(0) or division by 0
        b_pct = np.clip(b_pct, epsilon, 1.0)
        t_pct = np.clip(t_pct, epsilon, 1.0)

        # Normalize fractions so they sum to 1
        b_pct = b_pct / np.sum(b_pct)
        t_pct = t_pct / np.sum(t_pct)

        # PSI formula: sum((target% - baseline%) * ln(target% / baseline%))
        psi_val = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
        return float(max(0.0, psi_val))

    def detect_feature_drift(
        self,
        feature_name: str,
        group: FeatureGroup,
        current_records: Optional[List[FeatureRecord]] = None,
        psi_threshold: float = 0.25,
    ) -> FeatureDriftReport:
        """Evaluate distribution drift for a single feature between baseline store and current data."""
        # Baseline from historical store
        baseline_recs = self.store.read_records(group=group)
        b_vals: List[float] = []
        for r in baseline_recs:
            v = r.features.get(feature_name)
            if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                b_vals.append(float(v))

        # Current records from argument or pipeline sliding window
        target_recs: List[FeatureRecord] = []
        if current_records is not None:
            target_recs = current_records
        elif self.pipeline is not None:
            target_recs = [r for r in self.pipeline._window_history if r.feature_group == group]

        t_vals: List[float] = []
        for r in target_recs:
            v = r.features.get(feature_name)
            if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                t_vals.append(float(v))

        if len(b_vals) < 3 or len(t_vals) < 3:
            return FeatureDriftReport(
                feature_name=feature_name,
                feature_group=group,
                psi_score=0.0,
                baseline_mean=float(np.mean(b_vals)) if b_vals else None,
                target_mean=float(np.mean(t_vals)) if t_vals else None,
                drift_detected=False,
                threshold=psi_threshold,
                severity=AlertSeverity.INFO,
                details={"reason": "insufficient_sample_size", "baseline_n": len(b_vals), "target_n": len(t_vals)},
            )

        b_arr = np.array(b_vals, dtype=float)
        t_arr = np.array(t_vals, dtype=float)

        psi = self.calculate_psi(baseline=b_arr, target=t_arr)
        b_mean = float(np.mean(b_arr))
        t_mean = float(np.mean(t_arr))

        drift_detected = psi >= psi_threshold
        if psi >= 0.35:
            severity = AlertSeverity.CRITICAL
        elif psi >= psi_threshold:
            severity = AlertSeverity.WARNING
        elif psi >= 0.10:
            severity = AlertSeverity.INFO
        else:
            severity = AlertSeverity.INFO

        return FeatureDriftReport(
            feature_name=feature_name,
            feature_group=group,
            psi_score=round(psi, 4),
            baseline_mean=round(b_mean, 4),
            target_mean=round(t_mean, 4),
            drift_detected=drift_detected,
            threshold=psi_threshold,
            severity=severity,
            details={
                "baseline_count": len(b_vals),
                "target_count": len(t_vals),
                "baseline_std": round(float(np.std(b_arr)), 4),
                "target_std": round(float(np.std(t_arr)), 4),
            },
        )

    def check_all_features_drift(
        self,
        group: Optional[FeatureGroup] = None,
        current_records: Optional[List[FeatureRecord]] = None,
        psi_threshold: float = 0.25,
    ) -> List[FeatureDriftReport]:
        """Detect drift across all numeric features within target or all groups."""
        target_groups = [group] if group is not None else list(FeatureGroup)
        reports: List[FeatureDriftReport] = []

        for grp in target_groups:
            # Discover available features from store parquet partition
            fp = self.store._get_parquet_path(grp)
            if not fp.exists():
                continue

            import pyarrow.parquet as pq

            schema = pq.read_schema(fp)
            for name in schema.names:
                if name.startswith("feat_"):
                    feat_name = name[5:]
                    rep = self.detect_feature_drift(
                        feature_name=feat_name,
                        group=grp,
                        current_records=current_records,
                        psi_threshold=psi_threshold,
                    )
                    reports.append(rep)

        return reports

    def generate_telemetry_alerts(
        self,
        collector: Optional[RiskTelemetryCollector] = None,
        reports: Optional[List[FeatureDriftReport]] = None,
        psi_threshold: float = 0.25,
    ) -> List[TelemetryAlert]:
        """Generate and dispatch TelemetryAlert instances for all drifting features."""
        import uuid

        active_reports = (
            reports
            if reports is not None
            else self.check_all_features_drift(psi_threshold=psi_threshold)
        )
        alerts: List[TelemetryAlert] = []

        for rep in active_reports:
            if rep.drift_detected:
                alert = TelemetryAlert(
                    alert_id=f"alert-drift-{rep.feature_group.value.lower()}-{uuid.uuid4().hex[:8]}",
                    severity=rep.severity,
                    metric_name=f"{rep.feature_group.value}.{rep.feature_name}.psi",
                    threshold=rep.threshold,
                    observed_value=rep.psi_score,
                    message=(
                        f"Feature drift detected: PSI {rep.psi_score} >= {rep.threshold} "
                        f"(baseline mean: {rep.baseline_mean}, target mean: {rep.target_mean})"
                    ),
                )
                if collector is not None:
                    for listener in collector._alert_listeners:
                        try:
                            listener(alert)
                        except Exception:
                            pass
                alerts.append(alert)

        return alerts
