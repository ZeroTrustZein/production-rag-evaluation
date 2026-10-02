"""Parquet-backed columnar feature store engine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

from production_rag_evaluation.features.models import (
    FeatureGroup,
    FeatureRecord,
)


class ParquetFeatureStore:
    """Columnar feature store providing durable Parquet storage and retrieval."""

    def __init__(self, base_path: Union[str, Path] = "./data/feature_store") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_group_dir(self, group: FeatureGroup) -> Path:
        group_dir = self.base_path / group.value.lower()
        group_dir.mkdir(parents=True, exist_ok=True)
        return group_dir

    def _get_parquet_path(self, group: FeatureGroup) -> Path:
        return self._get_group_dir(group) / "features.parquet"

    def write_records(
        self,
        records: List[FeatureRecord],
        compression: str = "snappy",
    ) -> int:
        """Persist a batch of feature records to columnar Parquet files partitioned by group."""
        if not records:
            return 0

        # Group records by feature group
        grouped: Dict[FeatureGroup, List[FeatureRecord]] = {}
        for rec in records:
            grouped.setdefault(rec.feature_group, []).append(rec)

        written_count = 0
        for group, recs in grouped.items():
            file_path = self._get_parquet_path(group)

            # Build row dicts
            rows: List[Dict[str, Any]] = []
            for r in recs:
                row_data: Dict[str, Any] = {
                    "entity_id": r.entity_id,
                    "feature_group": r.feature_group.value,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": json.dumps(r.metadata),
                }
                for f_name, f_val in r.features.items():
                    row_data[f"feat_{f_name}"] = f_val
                rows.append(row_data)

            new_table = pa.Table.from_pylist(rows)

            if file_path.exists():
                existing_table = pq.read_table(file_path)
                combined_table = pa.concat_tables([existing_table, new_table], promote_options="default")
                pq.write_table(combined_table, file_path, compression=compression)
            else:
                pq.write_table(new_table, file_path, compression=compression)

            written_count += len(recs)

        return written_count

    def read_records(
        self,
        group: FeatureGroup,
        entity_id: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[FeatureRecord]:
        """Read and reconstruct FeatureRecord instances from the Parquet partition."""
        file_path = self._get_parquet_path(group)
        if not file_path.exists():
            return []

        table = pq.read_table(file_path)
        data = table.to_pylist()

        records: List[FeatureRecord] = []
        for row in data:
            if entity_id is not None and row.get("entity_id") != entity_id:
                continue

            features: Dict[str, Any] = {}
            for k, v in row.items():
                if k.startswith("feat_"):
                    feat_name = k[5:]
                    if columns is None or feat_name in columns:
                        features[feat_name] = v

            meta_str = row.get("metadata", "{}")
            try:
                meta = json.loads(meta_str) if isinstance(meta_str, str) else {}
            except Exception:
                meta = {}

            ts_raw = row.get("timestamp")
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except Exception:
                    ts = datetime.now(timezone.utc)
            elif isinstance(ts_raw, datetime):
                ts = ts_raw
            else:
                ts = datetime.now(timezone.utc)

            rec = FeatureRecord(
                entity_id=row["entity_id"],
                feature_group=FeatureGroup(row["feature_group"]),
                features=features,
                timestamp=ts,
                metadata=meta,
            )
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break

        return records

    def count_records(self, group: Optional[FeatureGroup] = None) -> int:
        """Count total persisted records across all or a specific group."""
        groups_to_check = [group] if group else list(FeatureGroup)
        total = 0
        for grp in groups_to_check:
            fp = self._get_parquet_path(grp)
            if fp.exists():
                meta = pq.read_metadata(fp)
                total += meta.num_rows
        return total

    def storage_summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics on the stored Parquet tables."""
        summary: Dict[str, Any] = {"base_path": str(self.base_path), "groups": {}}
        total_bytes = 0
        total_rows = 0

        for grp in FeatureGroup:
            fp = self._get_parquet_path(grp)
            if fp.exists():
                size = fp.stat().st_size
                meta = pq.read_metadata(fp)
                summary["groups"][grp.value] = {
                    "num_rows": meta.num_rows,
                    "num_columns": meta.num_columns,
                    "size_bytes": size,
                    "num_row_groups": meta.num_row_groups,
                }
                total_bytes += size
                total_rows += meta.num_rows

        summary["total_bytes"] = total_bytes
        summary["total_records"] = total_rows
        return summary
