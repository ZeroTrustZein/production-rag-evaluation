"""Real-time streaming feature extraction pipeline."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union

from production_rag_evaluation.features.extractors import (
    BaseFeatureExtractor,
    ChunkFeatureExtractor,
    EvaluationFeatureExtractor,
    QueryFeatureExtractor,
    ResponseFeatureExtractor,
)
from production_rag_evaluation.features.models import (
    FeatureGroup,
    FeatureRecord,
    StreamingBatchStats,
)
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    EvaluationMetrics,
    RAGQuery,
    RAGResponse,
)


class StreamingFeaturePipeline:
    """Orchestrates real-time streaming feature extraction and columnar Parquet buffering."""

    def __init__(
        self,
        store: Optional[ParquetFeatureStore] = None,
        buffer_size: int = 100,
        auto_flush: bool = True,
    ) -> None:
        self.store = store or ParquetFeatureStore()
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush

        # Default extractor suite
        self.query_extractor = QueryFeatureExtractor()
        self.chunk_extractor = ChunkFeatureExtractor()
        self.response_extractor = ResponseFeatureExtractor()
        self.evaluation_extractor = EvaluationFeatureExtractor()

        # In-memory buffer
        self._buffer: List[FeatureRecord] = []
        self._total_ingested: int = 0
        self._total_extracted: int = 0

    @property
    def buffer_count(self) -> int:
        """Current number of unflushed records in buffer."""
        return len(self._buffer)

    def ingest(self, item: Any) -> Optional[FeatureRecord]:
        """Polymorphically ingest a RAG event, extract features, and buffer."""
        self._total_ingested += 1
        record: Optional[FeatureRecord] = None

        if isinstance(item, RAGQuery):
            record = self.query_extractor.extract(item)
        elif isinstance(item, DocumentChunk):
            record = self.chunk_extractor.extract(item)
        elif isinstance(item, RAGResponse):
            record = self.response_extractor.extract(item)
        elif isinstance(item, EvaluationMetrics):
            record = self.evaluation_extractor.extract(item)
        elif isinstance(item, FeatureRecord):
            record = item

        if record is not None:
            self._buffer.append(record)
            self._total_extracted += 1

            if self.auto_flush and len(self._buffer) >= self.buffer_size:
                self.flush()

        return record

    def ingest_batch(self, items: List[Any]) -> List[FeatureRecord]:
        """Ingest a collection of heterogeneous items."""
        records: List[FeatureRecord] = []
        for it in items:
            rec = self.ingest(it)
            if rec is not None:
                records.append(rec)
        return records

    def flush(self) -> StreamingBatchStats:
        """Persist all unflushed feature records to Parquet columnar storage."""
        start_time = time.perf_counter()
        batch_id = str(uuid.uuid4())
        count_to_write = len(self._buffer)

        if count_to_write == 0:
            return StreamingBatchStats(
                batch_id=batch_id,
                records_ingested=0,
                features_materialized=0,
                processing_time_ms=0.0,
                storage_flush_success=True,
                error_count=0,
            )

        records_to_flush = list(self._buffer)
        self._buffer.clear()

        error_count = 0
        success = True
        try:
            self.store.write_records(records_to_flush)
        except Exception:
            # On failure, return to buffer
            self._buffer.extend(records_to_flush)
            success = False
            error_count = count_to_write

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return StreamingBatchStats(
            batch_id=batch_id,
            records_ingested=count_to_write,
            features_materialized=count_to_write if success else 0,
            processing_time_ms=round(elapsed_ms, 2),
            storage_flush_success=success,
            error_count=error_count,
        )

    def stats(self) -> Dict[str, Any]:
        """Return operational pipeline statistics."""
        return {
            "buffer_count": len(self._buffer),
            "buffer_capacity": self.buffer_size,
            "total_ingested": self._total_ingested,
            "total_extracted": self._total_extracted,
            "auto_flush": self.auto_flush,
            "store_summary": self.store.storage_summary(),
        }
