"""Data models, schemas, and typed records for the feature store."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FeatureValueType(str, Enum):
    """Primitive and structured data types for feature values."""

    FLOAT = "FLOAT"
    INT = "INT"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "TIMESTAMP"
    LIST_FLOAT = "LIST_FLOAT"


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
    features: List[FeatureDefinition] = Field(default_factory=list, description="List of feature specifications")
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

    entity_id: str = Field(description="Primary entity identifier (e.g., query_id, chunk_id, eval_id)")
    feature_group: FeatureGroup = Field(description="Feature group namespace")
    features: Dict[str, Any] = Field(default_factory=dict, description="Key-value mapping of extracted features")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when features were calculated or recorded",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operational or source metadata")


class StreamingBatchStats(BaseModel):
    """Metrics and diagnostics from a streaming feature batch processing cycle."""

    batch_id: str = Field(description="Unique batch identifier")
    records_ingested: int = Field(ge=0, description="Count of raw records processed")
    features_materialized: int = Field(ge=0, description="Count of feature records emitted")
    processing_time_ms: float = Field(ge=0.0, description="Elapsed execution time in milliseconds")
    storage_flush_success: bool = Field(default=True, description="Whether Parquet persistence succeeded")
    error_count: int = Field(default=0, ge=0, description="Number of extraction errors encountered")
