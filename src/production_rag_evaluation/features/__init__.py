"""Streaming feature extraction pipeline and Parquet columnar storage engine."""

from __future__ import annotations

from production_rag_evaluation.features.extractors import (
    BaseFeatureExtractor,
    ChunkFeatureExtractor,
    EvaluationFeatureExtractor,
    QueryFeatureExtractor,
    ResponseFeatureExtractor,
)
from production_rag_evaluation.features.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureRecord,
    FeatureSchema,
    FeatureValueType,
    StreamingBatchStats,
)
from production_rag_evaluation.features.pipeline import StreamingFeaturePipeline
from production_rag_evaluation.features.store import ParquetFeatureStore

__all__ = [
    # Schemas & Models
    "FeatureDefinition",
    "FeatureGroup",
    "FeatureRecord",
    "FeatureSchema",
    "FeatureValueType",
    "StreamingBatchStats",
    # Extractors
    "BaseFeatureExtractor",
    "QueryFeatureExtractor",
    "ChunkFeatureExtractor",
    "ResponseFeatureExtractor",
    "EvaluationFeatureExtractor",
    # Engines & Pipelines
    "ParquetFeatureStore",
    "StreamingFeaturePipeline",
]
