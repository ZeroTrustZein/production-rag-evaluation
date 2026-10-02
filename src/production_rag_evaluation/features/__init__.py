"""Streaming feature extraction pipeline and Parquet columnar storage engine."""

from __future__ import annotations

from production_rag_evaluation.features.extractors import (
    BaseFeatureExtractor,
    ChunkFeatureExtractor,
    EvaluationFeatureExtractor,
    FinancialFeatureExtractor,
    QueryFeatureExtractor,
    ResponseFeatureExtractor,
)
from production_rag_evaluation.features.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureRecord,
    FeatureSchema,
    FeatureStatistics,
    FeatureValueType,
    OnlineFeatureVector,
    SlidingWindowStats,
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
    "FeatureStatistics",
    "FeatureValueType",
    "OnlineFeatureVector",
    "SlidingWindowStats",
    "StreamingBatchStats",
    # Extractors
    "BaseFeatureExtractor",
    "QueryFeatureExtractor",
    "ChunkFeatureExtractor",
    "ResponseFeatureExtractor",
    "EvaluationFeatureExtractor",
    "FinancialFeatureExtractor",
    # Engines & Pipelines
    "ParquetFeatureStore",
    "StreamingFeaturePipeline",
]

