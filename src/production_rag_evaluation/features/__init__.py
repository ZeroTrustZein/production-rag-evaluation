"""Streaming feature extraction pipeline and Parquet columnar storage engine."""

from __future__ import annotations

from production_rag_evaluation.features.drift import FeatureDriftDetector
from production_rag_evaluation.features.extractors import (
    BaseFeatureExtractor,
    ChunkFeatureExtractor,
    EvaluationFeatureExtractor,
    FinancialFeatureExtractor,
    QueryFeatureExtractor,
    ResponseFeatureExtractor,
)
from production_rag_evaluation.features.handlers import (
    EvaluationFeatureHandler,
    FeatureCompactionHandler,
    FeatureExportHandler,
)
from production_rag_evaluation.features.models import (
    CompactionResult,
    ExportFormat,
    FeatureDefinition,
    FeatureDriftReport,
    FeatureGroup,
    FeatureRecord,
    FeatureSchema,
    FeatureStatistics,
    FeatureValueType,
    OnlineFeatureVector,
    SchemaValidationResult,
    SlidingWindowStats,
    StreamingBatchStats,
)
from production_rag_evaluation.features.pipeline import StreamingFeaturePipeline
from production_rag_evaluation.features.registry import (
    FeatureRegistry,
    get_default_registry,
)
from production_rag_evaluation.features.store import ParquetFeatureStore

__all__ = [
    # Schemas & Models
    "CompactionResult",
    "ExportFormat",
    "FeatureDefinition",
    "FeatureDriftReport",
    "FeatureGroup",
    "FeatureRecord",
    "FeatureSchema",
    "FeatureStatistics",
    "FeatureValueType",
    "OnlineFeatureVector",
    "SchemaValidationResult",
    "SlidingWindowStats",
    "StreamingBatchStats",
    # Extractors
    "BaseFeatureExtractor",
    "QueryFeatureExtractor",
    "ChunkFeatureExtractor",
    "ResponseFeatureExtractor",
    "EvaluationFeatureExtractor",
    "FinancialFeatureExtractor",
    # Registry & Validation
    "FeatureRegistry",
    "get_default_registry",
    # Handlers & Integrations
    "EvaluationFeatureHandler",
    "FeatureCompactionHandler",
    "FeatureExportHandler",
    "FeatureDriftDetector",
    # Engines & Pipelines
    "ParquetFeatureStore",
    "StreamingFeaturePipeline",
]
