"""Streaming feature extractors for financial RAG pipelines."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict

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
