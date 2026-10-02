"""Integration handlers, pipeline bridges, compaction, and dataset export."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

from production_rag_evaluation.features.models import (
    CompactionResult,
    ExportFormat,
    FeatureGroup,
    FeatureRecord,
)
from production_rag_evaluation.features.pipeline import StreamingFeaturePipeline
from production_rag_evaluation.features.store import ParquetFeatureStore
from production_rag_evaluation.finance.telemetry import RiskTelemetryCollector
from production_rag_evaluation.models.schemas import (
    EvaluationMetrics,
    RAGQuery,
    RAGResponse,
    RiskMetrics,
)


class EvaluationFeatureHandler:
    """Subsystem integration handler linking evaluation lifecycles to streaming feature stores."""

    def __init__(
        self,
        pipeline: Optional[StreamingFeaturePipeline] = None,
        auto_flush: bool = False,
        telemetry_collector: Optional[RiskTelemetryCollector] = None,
    ) -> None:
        self.pipeline = pipeline or StreamingFeaturePipeline()
        self.auto_flush = auto_flush
        self.telemetry_collector = telemetry_collector

    def handle_evaluation(
        self,
        query: RAGQuery,
        response: RAGResponse,
        metrics: EvaluationMetrics,
        flush_immediately: bool = False,
    ) -> List[FeatureRecord]:
        """Ingest all artifacts from a single RAG evaluation cycle into the feature store."""
        extracted: List[FeatureRecord] = []

        # 1. Ingest Query
        q_rec = self.pipeline.ingest(query)
        if q_rec:
            extracted.append(q_rec)

        # 2. Ingest Retrieved Chunks
        for chunk in response.retrieved_chunks:
            c_rec = self.pipeline.ingest(chunk)
            if c_rec:
                extracted.append(c_rec)

        # 3. Ingest Response
        r_rec = self.pipeline.ingest(response)
        if r_rec:
            extracted.append(r_rec)

        # 4. Ingest Evaluation Metrics
        e_rec = self.pipeline.ingest(metrics)
        if e_rec:
            extracted.append(e_rec)

        # 5. Bridge to telemetry collector if configured
        if self.telemetry_collector is not None:
            self.telemetry_collector.record_evaluation(metrics)

        # 6. Flush if requested or auto_flush is active
        if flush_immediately or self.auto_flush:
            self.pipeline.flush()

        return extracted

    def handle_batch_evaluations(
        self,
        pairs: List[Tuple[RAGQuery, RAGResponse]],
        metrics_list: List[EvaluationMetrics],
        flush_immediately: bool = True,
    ) -> List[FeatureRecord]:
        """Process an entire batch of evaluations and materialize features."""
        all_extracted: List[FeatureRecord] = []
        for (q, r), m in zip(pairs, metrics_list):
            records = self.handle_evaluation(query=q, response=r, metrics=m, flush_immediately=False)
            all_extracted.extend(records)

        if flush_immediately or self.auto_flush:
            self.pipeline.flush()

        return all_extracted

    def handle_risk_metrics(
        self,
        risk: RiskMetrics,
        entity_id: Optional[str] = None,
        flush_immediately: bool = False,
    ) -> Optional[FeatureRecord]:
        """Ingest quantitative portfolio or trading risk indicators."""
        rec = self.pipeline.ingest(risk)
        if rec and entity_id:
            rec = FeatureRecord(
                entity_id=entity_id,
                feature_group=rec.feature_group,
                features=rec.features,
                timestamp=rec.timestamp,
                metadata=rec.metadata,
            )
            self.pipeline.store.put_online_record(rec)

        if flush_immediately or self.auto_flush:
            self.pipeline.flush()

        return rec

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Retrieve live telemetry and feature store health stats."""
        stats = self.pipeline.stats()
        if self.telemetry_collector is not None:
            stats["collector_summary"] = self.telemetry_collector.export_summary()
        return stats


class FeatureCompactionHandler:
    """Manages Parquet table compaction, deduplication, and retention pruning."""

    def __init__(self, store: ParquetFeatureStore) -> None:
        self.store = store

    def compact_group(
        self,
        group: FeatureGroup,
        ttl_seconds: Optional[float] = None,
        compression: str = "snappy",
    ) -> CompactionResult:
        """Consolidate fragmented parquet files, remove TTL-expired rows, and rewrite optimized table."""
        start_time = time.perf_counter()
        file_path = self.store._get_parquet_path(group)

        if not file_path.exists():
            return CompactionResult(
                group=group,
                records_compacted=0,
                bytes_before=0,
                bytes_after=0,
                compression_ratio=1.0,
                expired_records_pruned=0,
                duration_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        bytes_before = file_path.stat().st_size
        records = self.store.read_records(group=group)

        # Filter out expired records if TTL is set
        expired_count = 0
        if ttl_seconds is not None:
            now_utc = datetime.now(timezone.utc)
            surviving: List[FeatureRecord] = []
            for r in records:
                r_ts = r.timestamp if r.timestamp.tzinfo is not None else r.timestamp.replace(tzinfo=timezone.utc)
                age = (now_utc - r_ts).total_seconds()
                if age <= ttl_seconds:
                    surviving.append(r)
                else:
                    expired_count += 1
            records = surviving

        # Rewrite clean compacted parquet file
        if records:
            rows: List[Dict[str, Any]] = []
            for r in records:
                row_data: Dict[str, Any] = {
                    "entity_id": r.entity_id,
                    "feature_group": r.feature_group.value,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": json.dumps(r.metadata),
                }
                for f_name, f_val in r.features.items():
                    row_data[f"feat_{f_name}"] = f_val
                rows.append(row_data)

            table = pa.Table.from_pylist(rows)
            pq.write_table(table, file_path, compression=compression)
            bytes_after = file_path.stat().st_size
        else:
            # All expired or zero records
            if file_path.exists():
                file_path.unlink()
            bytes_after = 0

        # Refresh online cache for surviving records
        for r in records:
            self.store.put_online_record(r)

        compression_ratio = round(bytes_after / bytes_before, 4) if bytes_before > 0 else 1.0
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return CompactionResult(
            group=group,
            records_compacted=len(records),
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            compression_ratio=compression_ratio,
            expired_records_pruned=expired_count,
            duration_ms=elapsed_ms,
        )

    def compact_all(
        self,
        ttl_seconds: Optional[float] = None,
        compression: str = "snappy",
    ) -> Dict[FeatureGroup, CompactionResult]:
        """Perform storage compaction across all feature group partitions."""
        results: Dict[FeatureGroup, CompactionResult] = {}
        for group in FeatureGroup:
            results[group] = self.compact_group(
                group=group, ttl_seconds=ttl_seconds, compression=compression
            )
        return results

    def prune_expired_records(self, group: FeatureGroup, older_than: datetime) -> int:
        """Purge records strictly older than the specified timestamp."""
        now_utc = datetime.now(timezone.utc)
        target_utc = older_than if older_than.tzinfo is not None else older_than.replace(tzinfo=timezone.utc)
        ttl = max(0.0, (now_utc - target_utc).total_seconds())
        res = self.compact_group(group=group, ttl_seconds=ttl)
        return res.expired_records_pruned


class FeatureExportHandler:
    """Subsystem exporter generating aligned point-in-time datasets for offline training & auditing."""

    def __init__(self, store: ParquetFeatureStore) -> None:
        self.store = store

    def export_dataset(
        self,
        entity_ids: List[str],
        output_path: Optional[Union[str, Path]] = None,
        format: ExportFormat = ExportFormat.JSONL,
        as_of: Optional[datetime] = None,
        groups: Optional[List[FeatureGroup]] = None,
    ) -> Union[List[Dict[str, Any]], Path]:
        """Construct point-in-time multi-group feature matrix and export to chosen format."""
        matrix = self.store.get_historical_matrix(
            entity_ids=entity_ids, groups=groups, as_of=as_of
        )

        if output_path is None or format == ExportFormat.DICT:
            return matrix

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ExportFormat.JSONL:
            with open(out_path, "w", encoding="utf-8") as f:
                for row in matrix:
                    f.write(json.dumps(row) + "\n")
        elif format == ExportFormat.CSV:
            if not matrix:
                out_path.write_text("", encoding="utf-8")
            else:
                fieldnames = sorted({k for row in matrix for k in row.keys()})
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in matrix:
                        writer.writerow(row)
        elif format == ExportFormat.PARQUET:
            if matrix:
                table = pa.Table.from_pylist(matrix)
                pq.write_table(table, out_path)
            else:
                empty_schema = pa.schema([pa.field("entity_id", pa.string())])
                pq.write_table(pa.Table.from_pylist([], schema=empty_schema), out_path)

        return out_path
