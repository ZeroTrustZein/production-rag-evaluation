"""Data models, schemas, and typed records for the feature store."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from production_rag_evaluation.models.schemas import AlertSeverity


def ensure_utc(val: Any) -> datetime:
    """Normalize datetime or ISO string to UTC-aware datetime."""
    if isinstance(val, str):
        try:
            val = datetime.fromisoformat(val)
        except Exception:
            return datetime.now(timezone.utc)
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


class ExportFormat(str, Enum):
    """Supported output formats for dataset exports."""

    JSONL = "JSONL"
    PARQUET = "PARQUET"
    CSV = "CSV"
    DICT = "DICT"


class SchemaValidationResult(BaseModel):
    """Result of validating a feature record against a schema."""

    model_config = ConfigDict(frozen=True)

    valid: bool = Field(description="Whether the record adheres strictly to the schema")
    errors: List[str] = Field(default_factory=list, description="Validation failure descriptions")
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings (e.g. unmapped extra fields)"
    )


class CompactionResult(BaseModel):
    """Metrics from compacting and purging a Parquet group partition."""

    model_config = ConfigDict(frozen=True)

    group: FeatureGroup = Field(description="Partition feature group")
    records_compacted: int = Field(ge=0, description="Number of active records consolidated")
    bytes_before: int = Field(ge=0, description="Storage size in bytes before compaction")
    bytes_after: int = Field(ge=0, description="Storage size in bytes after compaction")
    compression_ratio: float = Field(
        ge=0.0, default=1.0, description="Ratio of bytes_after to bytes_before"
    )
    expired_records_pruned: int = Field(
        ge=0, default=0, description="Count of TTL-expired records dropped"
    )
    duration_ms: float = Field(ge=0.0, default=0.0, description="Time taken in milliseconds")


class FeatureDriftReport(BaseModel):
    """Report detailing detected distribution drift for a single feature."""

    model_config = ConfigDict(frozen=True)

    feature_name: str = Field(description="Name of evaluated feature")
    feature_group: FeatureGroup = Field(description="Feature group namespace")
    psi_score: float = Field(ge=0.0, description="Population Stability Index (PSI)")
    baseline_mean: Optional[float] = Field(
        default=None, description="Mean of historical baseline distribution"
    )
    target_mean: Optional[float] = Field(
        default=None, description="Mean of current/window distribution"
    )
    drift_detected: bool = Field(default=False, description="Whether PSI exceeded threshold")
    threshold: float = Field(default=0.25, ge=0.0, description="PSI alert threshold applied")
    severity: AlertSeverity = Field(
        default=AlertSeverity.INFO, description="Alert severity classification"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Bucket-level drift diagnostics"
    )


class FeatureValueType(str, Enum):
    """Primitive and structured data types for feature values."""

    FLOAT = "FLOAT"
    INT = "INT"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "TIMESTAMP"
    LIST_FLOAT = "LIST_FLOAT"

    def is_valid(self, val: Any) -> bool:
        """Validate whether a given runtime value conforms to this feature value type."""
        if val is None:
            return False
        if self == FeatureValueType.BOOLEAN:
            return isinstance(val, bool)
        if self == FeatureValueType.INT:
            return isinstance(val, int) and not isinstance(val, bool)
        if self == FeatureValueType.FLOAT:
            return isinstance(val, (int, float)) and not isinstance(val, bool)
        if self == FeatureValueType.STRING:
            return isinstance(val, str)
        if self == FeatureValueType.TIMESTAMP:
            return isinstance(val, (datetime, str))
        if self == FeatureValueType.LIST_FLOAT:
            return isinstance(val, (list, tuple, np.ndarray)) and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in val
            )
        return False


class FeatureGroup(str, Enum):
    """Domain grouping for feature sets in the RAG pipeline."""

    QUERY = "QUERY"
    CHUNK = "CHUNK"
    RESPONSE = "RESPONSE"
    EVALUATION = "EVALUATION"
    FINANCIAL = "FINANCIAL"


class FeatureDefinition(BaseModel):
    """Specification of an individual feature."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique identifier for the feature within its group")
    feature_group: FeatureGroup = Field(description="Group/namespace of the feature")
    value_type: FeatureValueType = Field(description="Data type of the feature value")
    description: str = Field(default="", description="Detailed human-readable documentation")
    nullable: bool = Field(default=True, description="Whether the feature can contain None")
    default_value: Optional[Any] = Field(default=None, description="Fallback value if missing")


class FeatureSchema(BaseModel):
    """Schema definition for a group of features."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Schema or table name")
    feature_group: FeatureGroup = Field(description="Target feature group")
    features: List[FeatureDefinition] = Field(
        default_factory=list, description="List of feature specifications"
    )
    version: str = Field(default="1.0.0", description="Schema version identifier")

    def get_feature(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Look up a feature definition by name."""
        for feat in self.features:
            if feat.name == feature_name:
                return feat
        return None

    def names(self) -> List[str]:
        """List all feature names in the schema."""
        return [f.name for f in self.features]


class FeatureRecord(BaseModel):
    """A single materialized feature observation for an entity."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(
        description="Primary entity identifier (e.g., query_id, chunk_id, eval_id)"
    )
    feature_group: FeatureGroup = Field(description="Feature group namespace")
    features: Dict[str, Any] = Field(
        default_factory=dict, description="Key-value mapping of extracted features"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when features were calculated or recorded",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Operational or source metadata"
    )


class StreamingBatchStats(BaseModel):
    """Metrics and diagnostics from a streaming feature batch processing cycle."""

    batch_id: str = Field(description="Unique batch identifier")
    records_ingested: int = Field(ge=0, description="Count of raw records processed")
    features_materialized: int = Field(ge=0, description="Count of feature records emitted")
    processing_time_ms: float = Field(ge=0.0, description="Elapsed execution time in milliseconds")
    storage_flush_success: bool = Field(
        default=True, description="Whether Parquet persistence succeeded"
    )
    error_count: int = Field(default=0, ge=0, description="Number of extraction errors encountered")


class FeatureStatistics(BaseModel):
    """Statistical summary and distribution metrics for a single feature column."""

    model_config = ConfigDict(frozen=True)

    feature_name: str = Field(description="Feature identifier")
    feature_group: FeatureGroup = Field(description="Feature group namespace")
    count: int = Field(ge=0, description="Total observations analyzed")
    null_count: int = Field(ge=0, description="Number of null or missing values")
    mean: Optional[float] = Field(default=None, description="Arithmetic mean for numerical feature")
    std: Optional[float] = Field(default=None, description="Sample standard deviation")
    min_value: Optional[float] = Field(default=None, description="Minimum observed value")
    max_value: Optional[float] = Field(default=None, description="Maximum observed value")
    quantiles: Dict[str, float] = Field(
        default_factory=dict, description="Distribution quantiles (p25, p50, p75)"
    )


class SlidingWindowStats(BaseModel):
    """Rolling statistical aggregations over streaming feature events."""

    model_config = ConfigDict(frozen=True)

    window_size: int = Field(ge=1, description="Number of observations in the sliding window")
    count: int = Field(ge=0, description="Active count of elements in current window")
    avg_latency_ms: float = Field(
        ge=0.0, default=0.0, description="Mean latency across window events"
    )
    overall_pass_rate: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Proportion of passing evaluations"
    )
    avg_faithfulness: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Mean faithfulness score"
    )
    rolling_hallucination_rate: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Rolling hallucination frequency"
    )


class OnlineFeatureVector(BaseModel):
    """Low-latency feature vector for online real-time inference."""

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(description="Target entity identifier")
    feature_group: FeatureGroup = Field(description="Feature group namespace")
    features: Dict[str, Any] = Field(default_factory=dict, description="Feature key-values")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Freshness timestamp of the feature observation",
    )
