"""Streaming feature extractors for financial RAG pipelines."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

import numpy as np

from production_rag_evaluation.features.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureRecord,
    FeatureSchema,
    FeatureValueType,
)
from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    EvaluationMetrics,
    RAGQuery,
    RAGResponse,
    RiskMetrics,
)


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    def __init__(self, schema: FeatureSchema) -> None:
        self.schema = schema

    @property
    def feature_group(self) -> FeatureGroup:
        return self.schema.feature_group

    @abstractmethod
    def extract(self, item: Any) -> FeatureRecord:
        """Extract features from the incoming input item."""
        pass

    def validate(self, record: FeatureRecord) -> bool:
        """Validate an extracted FeatureRecord against the schema definition."""
        if record.feature_group != self.schema.feature_group:
            return False
        for feat_def in self.schema.features:
            if feat_def.name not in record.features:
                if not feat_def.nullable and feat_def.default_value is None:
                    return False
                continue
            val = record.features[feat_def.name]
            if val is None:
                if not feat_def.nullable:
                    return False
                continue
            if feat_def.value_type == FeatureValueType.FLOAT:
                if not isinstance(val, (float, int)) or isinstance(val, bool):
                    return False
            elif feat_def.value_type == FeatureValueType.INT:
                if not isinstance(val, int) or isinstance(val, bool):
                    return False
            elif feat_def.value_type == FeatureValueType.STRING:
                if not isinstance(val, str):
                    return False
            elif feat_def.value_type == FeatureValueType.BOOLEAN:
                if not isinstance(val, bool):
                    return False
            elif feat_def.value_type == FeatureValueType.LIST_FLOAT:
                if not isinstance(val, (list, tuple, np.ndarray)):
                    return False
        return True



class QueryFeatureExtractor(BaseFeatureExtractor):
    """Extracts lexical, financial, and structural features from queries."""

    FINANCIAL_TERMS = {
        "revenue", "sales", "margin", "ebitda", "net income", "eps",
        "guidance", "capex", "cash flow", "dividend", "debt", "growth"
    }

    def __init__(self) -> None:
        schema = FeatureSchema(
            name="rag_query_features",
            feature_group=FeatureGroup.QUERY,
            features=[
                FeatureDefinition(name="query_char_len", feature_group=FeatureGroup.QUERY, value_type=FeatureValueType.INT),
                FeatureDefinition(name="query_word_count", feature_group=FeatureGroup.QUERY, value_type=FeatureValueType.INT),
                FeatureDefinition(name="has_ticker", feature_group=FeatureGroup.QUERY, value_type=FeatureValueType.BOOLEAN),
                FeatureDefinition(name="financial_intent_density", feature_group=FeatureGroup.QUERY, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="has_numerical_target", feature_group=FeatureGroup.QUERY, value_type=FeatureValueType.BOOLEAN),
            ],
        )
        super().__init__(schema)

    def extract(self, query: RAGQuery) -> FeatureRecord:
        text = query.query_text
        words = text.split()
        word_count = len(words)
        char_len = len(text)
        has_ticker = bool(query.ticker)

        lower_text = text.lower()
        matched_terms = sum(1 for term in self.FINANCIAL_TERMS if term in lower_text)
        intent_density = float(matched_terms / max(1, word_count))

        num_matches = re.findall(r"\b\d+(?:\.\d+)?%?|\$\d+", text)
        has_num = len(num_matches) > 0

        features: Dict[str, Any] = {
            "query_char_len": char_len,
            "query_word_count": word_count,
            "has_ticker": has_ticker,
            "financial_intent_density": round(intent_density, 4),
            "has_numerical_target": has_num,
        }

        return FeatureRecord(
            entity_id=query.query_id,
            feature_group=FeatureGroup.QUERY,
            features=features,
            metadata={"ticker": query.ticker} if query.ticker else {},
        )


class ChunkFeatureExtractor(BaseFeatureExtractor):
    """Extracts lexical, numerical density, and metadata features from document chunks."""

    def __init__(self) -> None:
        schema = FeatureSchema(
            name="rag_chunk_features",
            feature_group=FeatureGroup.CHUNK,
            features=[
                FeatureDefinition(name="chunk_char_len", feature_group=FeatureGroup.CHUNK, value_type=FeatureValueType.INT),
                FeatureDefinition(name="chunk_word_count", feature_group=FeatureGroup.CHUNK, value_type=FeatureValueType.INT),
                FeatureDefinition(name="numerical_token_density", feature_group=FeatureGroup.CHUNK, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="doc_type_code", feature_group=FeatureGroup.CHUNK, value_type=FeatureValueType.STRING),
                FeatureDefinition(name="has_tables", feature_group=FeatureGroup.CHUNK, value_type=FeatureValueType.BOOLEAN),
            ],
        )
        super().__init__(schema)

    def extract(self, chunk: DocumentChunk) -> FeatureRecord:
        text = chunk.text
        words = text.split()
        word_count = len(words)
        char_len = len(text)

        num_tokens = re.findall(r"\$?\d+(?:\.\d+)?%?", text)
        density = float(len(num_tokens) / max(1, word_count))

        has_tables = ("|" in text or "\t" in text or "---" in text)

        features: Dict[str, Any] = {
            "chunk_char_len": char_len,
            "chunk_word_count": word_count,
            "numerical_token_density": round(density, 4),
            "doc_type_code": chunk.doc_type.value,
            "has_tables": has_tables,
        }

        return FeatureRecord(
            entity_id=chunk.chunk_id,
            feature_group=FeatureGroup.CHUNK,
            features=features,
            metadata={"ticker": chunk.ticker, "source": chunk.source_url},
        )


class ResponseFeatureExtractor(BaseFeatureExtractor):
    """Extracts grounding, citation ratio, and length features from answers."""

    def __init__(self) -> None:
        schema = FeatureSchema(
            name="rag_response_features",
            feature_group=FeatureGroup.RESPONSE,
            features=[
                FeatureDefinition(name="response_char_len", feature_group=FeatureGroup.RESPONSE, value_type=FeatureValueType.INT),
                FeatureDefinition(name="response_word_count", feature_group=FeatureGroup.RESPONSE, value_type=FeatureValueType.INT),
                FeatureDefinition(name="citation_count", feature_group=FeatureGroup.RESPONSE, value_type=FeatureValueType.INT),
                FeatureDefinition(name="citation_ratio", feature_group=FeatureGroup.RESPONSE, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="retrieved_chunk_count", feature_group=FeatureGroup.RESPONSE, value_type=FeatureValueType.INT),
            ],
        )
        super().__init__(schema)

    def extract(self, response: RAGResponse) -> FeatureRecord:
        text = response.generated_answer
        words = text.split()
        word_count = len(words)
        char_len = len(text)
        citation_count = len(response.citations)
        retrieved_count = len(response.retrieved_chunks)

        citation_ratio = float(citation_count / max(1, retrieved_count))

        features: Dict[str, Any] = {
            "response_char_len": char_len,
            "response_word_count": word_count,
            "citation_count": citation_count,
            "citation_ratio": round(citation_ratio, 4),
            "retrieved_chunk_count": retrieved_count,
        }

        return FeatureRecord(
            entity_id=response.query_id,
            feature_group=FeatureGroup.RESPONSE,
            features=features,
            metadata={"model_name": response.model_name, "latency_ms": response.latency_ms},
        )


class EvaluationFeatureExtractor(BaseFeatureExtractor):
    """Extracts structured evaluation scores and hallucination telemetry features."""

    def __init__(self) -> None:
        schema = FeatureSchema(
            name="rag_evaluation_features",
            feature_group=FeatureGroup.EVALUATION,
            features=[
                FeatureDefinition(name="faithfulness_score", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="answer_relevance_score", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="context_precision_score", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="context_recall_score", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="citation_grounding_score", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="hallucination_count", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.INT),
                FeatureDefinition(name="overall_pass", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.BOOLEAN),
                FeatureDefinition(name="evaluation_latency_ms", feature_group=FeatureGroup.EVALUATION, value_type=FeatureValueType.FLOAT),
            ],
        )
        super().__init__(schema)

    def extract(self, metrics: EvaluationMetrics) -> FeatureRecord:
        features: Dict[str, Any] = {
            "faithfulness_score": metrics.faithfulness_score,
            "answer_relevance_score": metrics.answer_relevance_score,
            "context_precision_score": metrics.context_precision_score,
            "context_recall_score": metrics.context_recall_score,
            "citation_grounding_score": metrics.citation_grounding_score,
            "hallucination_count": len(metrics.findings),
            "overall_pass": metrics.overall_pass,
            "evaluation_latency_ms": metrics.evaluation_latency_ms,
        }

        eval_id = str(metrics.metadata.get("evaluation_id", metrics.metadata.get("query_id", "eval_record")))
        return FeatureRecord(
            entity_id=eval_id,
            feature_group=FeatureGroup.EVALUATION,
            features=features,
            metadata=dict(metrics.metadata),
        )


class FinancialFeatureExtractor(BaseFeatureExtractor):
    """Extracts risk indicators and portfolio metrics into durable feature records."""

    def __init__(self) -> None:
        schema = FeatureSchema(
            name="rag_financial_features",
            feature_group=FeatureGroup.FINANCIAL,
            features=[
                FeatureDefinition(name="sharpe_ratio", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="sortino_ratio", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="max_drawdown", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="annualized_volatility", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="calmar_ratio", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="var_95", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="cvar_95", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.FLOAT),
                FeatureDefinition(name="risk_profile", feature_group=FeatureGroup.FINANCIAL, value_type=FeatureValueType.STRING),
            ],
        )
        super().__init__(schema)

    def extract(self, item: Any) -> FeatureRecord:
        if isinstance(item, RiskMetrics):
            metrics = item
        elif isinstance(item, (np.ndarray, list)):
            from production_rag_evaluation.finance.metrics import FinancialRiskEngine

            engine = FinancialRiskEngine()
            metrics = engine.compute_all_metrics(np.asarray(item, dtype=float))
        elif hasattr(item, "risk_metrics") and isinstance(item.risk_metrics, RiskMetrics):
            metrics = item.risk_metrics
        else:
            raise ValueError(f"Unsupported item type for FinancialFeatureExtractor: {type(item)}")

        features: Dict[str, Any] = {
            "sharpe_ratio": float(metrics.sharpe_ratio),
            "sortino_ratio": float(metrics.sortino_ratio),
            "max_drawdown": float(metrics.max_drawdown),
            "annualized_volatility": float(metrics.annualized_volatility),
            "calmar_ratio": float(metrics.calmar_ratio),
            "var_95": float(metrics.var_95),
            "cvar_95": float(metrics.cvar_95),
            "risk_profile": metrics.risk_profile.value,
        }

        entity_id = f"fin_{metrics.risk_profile.value.lower()}"
        return FeatureRecord(
            entity_id=entity_id,
            feature_group=FeatureGroup.FINANCIAL,
            features=features,
            metadata={"computed_at": datetime.now(timezone.utc).isoformat()},
        )

