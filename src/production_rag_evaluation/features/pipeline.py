"""Real-time streaming feature extraction pipeline."""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from production_rag_evaluation.features.extractors import (
    ChunkFeatureExtractor,
    EvaluationFeatureExtractor,
    FinancialFeatureExtractor,
    QueryFeatureExtractor,
    ResponseFeatureExtractor,
)
from production_rag_evaluation.features.models import (
    FeatureGroup,
    FeatureRecord,
    SlidingWindowStats,
    StreamingBatchStats,
)
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    EvaluationMetrics,
    RAGQuery,
    RAGResponse,
    RiskMetrics,
)


class StreamingFeaturePipeline:
    """Orchestrates real-time streaming feature extraction and columnar Parquet buffering."""

    def __init__(
        self,
        store: Optional[ParquetFeatureStore] = None,
        buffer_size: int = 100,
        auto_flush: bool = True,
        window_size: int = 50,
        deduplicate: bool = False,
    ) -> None:
        self.store = store or ParquetFeatureStore()
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        self.window_size = window_size
        self.deduplicate = deduplicate

        # Default extractor suite
        self.query_extractor = QueryFeatureExtractor()
        self.chunk_extractor = ChunkFeatureExtractor()
        self.response_extractor = ResponseFeatureExtractor()
        self.evaluation_extractor = EvaluationFeatureExtractor()
        self.financial_extractor = FinancialFeatureExtractor()

        # In-memory buffer & sliding window state
        self._buffer: List[FeatureRecord] = []
        self._window_history: Deque[FeatureRecord] = deque(maxlen=window_size)
        self._seen_entities: Set[Tuple[FeatureGroup, str]] = set()
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
        elif isinstance(item, RiskMetrics):
            record = self.financial_extractor.extract(item)
        elif isinstance(item, FeatureRecord):
            record = item

        if record is not None:
            # Check deduplication if enabled
            entity_key = (record.feature_group, record.entity_id)
            if self.deduplicate and entity_key in self._seen_entities:
                return record
            self._seen_entities.add(entity_key)

            # Synchronously update online feature store for real-time lookup
            self.store.put_online_record(record)

            # Record in sliding window
            self._window_history.append(record)

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

    def get_sliding_window_stats(self) -> SlidingWindowStats:
        """Compute rolling telemetry and quality metrics over the active sliding window."""
        count = len(self._window_history)
        if count == 0:
            return SlidingWindowStats(
                window_size=self.window_size,
                count=0,
                avg_latency_ms=0.0,
                overall_pass_rate=1.0,
                avg_faithfulness=1.0,
                rolling_hallucination_rate=0.0,
            )

        latencies: List[float] = []
        eval_records: List[FeatureRecord] = []

        for r in self._window_history:
            lat = r.features.get("evaluation_latency_ms") or r.metadata.get("latency_ms")
            if lat is not None and isinstance(lat, (int, float)):
                latencies.append(float(lat))
            if r.feature_group == FeatureGroup.EVALUATION:
                eval_records.append(r)

        avg_lat = float(sum(latencies) / len(latencies)) if latencies else 0.0

        if eval_records:
            pass_count = sum(1 for e in eval_records if e.features.get("overall_pass") is True)
            pass_rate = float(pass_count / len(eval_records))
            faithfulness_vals = [
                float(e.features.get("faithfulness_score", 1.0))
                for e in eval_records
                if "faithfulness_score" in e.features
            ]
            avg_faith = float(sum(faithfulness_vals) / len(faithfulness_vals)) if faithfulness_vals else 1.0
            hallucinated_count = sum(
                1 for e in eval_records if int(e.features.get("hallucination_count", 0)) > 0
            )
            hallucination_rate = float(hallucinated_count / len(eval_records))
        else:
            pass_rate = 1.0
            avg_faith = 1.0
            hallucination_rate = 0.0

        return SlidingWindowStats(
            window_size=self.window_size,
            count=count,
            avg_latency_ms=round(avg_lat, 2),
            overall_pass_rate=round(pass_rate, 4),
            avg_faithfulness=round(avg_faith, 4),
            rolling_hallucination_rate=round(hallucination_rate, 4),
        )


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
            "sliding_window": self.get_sliding_window_stats().model_dump(),
            "store_summary": self.store.storage_summary(),
        }

