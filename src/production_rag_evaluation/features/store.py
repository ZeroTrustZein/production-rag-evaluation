"""Parquet-backed columnar feature store engine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from production_rag_evaluation.features.models import (
    FeatureGroup,
    FeatureRecord,
    FeatureStatistics,
    ensure_utc,
)


class ParquetFeatureStore:
    """Columnar feature store providing durable Parquet storage, online serving, and point-in-time joins."""

    def __init__(self, base_path: Union[str, Path] = "./data/feature_store") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._online_cache: Dict[Tuple[FeatureGroup, str], FeatureRecord] = {}

    def _get_group_dir(self, group: FeatureGroup) -> Path:
        group_dir = self.base_path / group.value.lower()
        group_dir.mkdir(parents=True, exist_ok=True)
        return group_dir

    def _get_parquet_path(self, group: FeatureGroup) -> Path:
        return self._get_group_dir(group) / "features.parquet"

    def put_online_record(self, record: FeatureRecord) -> None:
        """Cache a single feature record for low-latency online retrieval."""
        self._online_cache[(record.feature_group, record.entity_id)] = record

    def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
        group: Optional[FeatureGroup] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Retrieve latest feature vectors for a list of entity IDs with sub-millisecond latency."""
        results: Dict[str, Dict[str, Any]] = {}
        groups_to_check = [group] if group is not None else list(FeatureGroup)

        for eid in entity_ids:
            merged: Dict[str, Any] = {}
            for grp in groups_to_check:
                rec = self._online_cache.get((grp, eid))
                if rec is None:
                    # Fall back to disk if not in memory
                    disk_recs = self.read_records(group=grp, entity_id=eid)
                    if disk_recs:
                        rec = disk_recs[-1]
                        self.put_online_record(rec)
                if rec is not None:
                    for k, v in rec.features.items():
                        if feature_names is None or k in feature_names:
                            merged[k] = v
            results[eid] = merged
        return results

    def write_records(
        self,
        records: List[FeatureRecord],
        compression: str = "snappy",
    ) -> int:
        """Persist a batch of feature records to columnar Parquet files partitioned by group."""
        if not records:
            return 0

        # Update online in-memory cache
        for rec in records:
            self.put_online_record(rec)

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
                combined_table = pa.concat_tables(
                    [existing_table, new_table], promote_options="default"
                )
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

    def get_as_of(
        self,
        entity_id: str,
        as_of: datetime,
        group: Optional[FeatureGroup] = None,
    ) -> Optional[FeatureRecord]:
        """Retrieve the latest observation for an entity recorded on or before the given timestamp."""
        as_of_utc = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
        groups_to_check = [group] if group is not None else list(FeatureGroup)

        candidates: List[FeatureRecord] = []
        for grp in groups_to_check:
            # Check online cache first
            cached = self._online_cache.get((grp, entity_id))
            if cached is not None:
                c_ts = (
                    cached.timestamp
                    if cached.timestamp.tzinfo is not None
                    else cached.timestamp.replace(tzinfo=timezone.utc)
                )
                if c_ts <= as_of_utc:
                    candidates.append(cached)

            recs = self.read_records(group=grp, entity_id=entity_id)
            for r in recs:
                r_ts = (
                    r.timestamp
                    if r.timestamp.tzinfo is not None
                    else r.timestamp.replace(tzinfo=timezone.utc)
                )
                if r_ts <= as_of_utc:
                    candidates.append(r)

        if not candidates:
            return None

        candidates.sort(key=lambda r: r.timestamp)
        return candidates[-1]

    def get_historical_matrix(
        self,
        entity_ids: List[str],
        groups: Optional[List[FeatureGroup]] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Construct an aligned point-in-time multi-group feature matrix for entity evaluations."""
        target_groups = groups or list(FeatureGroup)
        matrix: List[Dict[str, Any]] = []

        for eid in entity_ids:
            row: Dict[str, Any] = {"entity_id": eid}
            has_data = False
            for grp in target_groups:
                rec: Optional[FeatureRecord] = None
                if as_of is not None:
                    rec = self.get_as_of(entity_id=eid, as_of=as_of, group=grp)
                else:
                    rec = self._online_cache.get((grp, eid))
                    if rec is None:
                        disk_recs = self.read_records(group=grp, entity_id=eid)
                        if disk_recs:
                            rec = disk_recs[-1]

                if rec is not None:
                    has_data = True
                    for feat_name, feat_val in rec.features.items():
                        row[f"{grp.value.lower()}__{feat_name}"] = feat_val
                    row[f"{grp.value.lower()}__timestamp"] = rec.timestamp.isoformat()

            if has_data:
                matrix.append(row)

        return matrix

    def compute_feature_statistics(self, group: FeatureGroup) -> Dict[str, FeatureStatistics]:
        """Calculate statistical summaries and distribution quantiles for features in a group."""
        file_path = self._get_parquet_path(group)
        if not file_path.exists():
            return {}

        table = pq.read_table(file_path)
        stats: Dict[str, FeatureStatistics] = {}

        for col in table.column_names:
            if not col.startswith("feat_"):
                continue
            feat_name = col[5:]
            vals = table.column(col).to_pylist()
            total_count = len(vals)
            non_null_vals = [v for v in vals if v is not None]
            null_count = total_count - len(non_null_vals)

            is_numeric = len(non_null_vals) > 0 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null_vals
            )

            if is_numeric:
                arr = np.array(non_null_vals, dtype=float)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                quantiles = {
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                }
                stats[feat_name] = FeatureStatistics(
                    feature_name=feat_name,
                    feature_group=group,
                    count=total_count,
                    null_count=null_count,
                    mean=round(mean_val, 4),
                    std=round(std_val, 4),
                    min_value=round(min_val, 4),
                    max_value=round(max_val, 4),
                    quantiles={k: round(v, 4) for k, v in quantiles.items()},
                )
            else:
                stats[feat_name] = FeatureStatistics(
                    feature_name=feat_name,
                    feature_group=group,
                    count=total_count,
                    null_count=null_count,
                )

        return stats
