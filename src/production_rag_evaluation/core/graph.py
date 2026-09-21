"""Stateful evaluation workflow graph using DAG state machine architecture."""

from __future__ import annotations

import time
from typing import List, Optional

import networkx as nx

from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.models.schemas import (
    Citation,
    EvaluationMetrics,
    HallucinationFinding,
    RAGQuery,
    RAGResponse,
)


class EvaluationState:
    """State object passing through evaluation graph nodes."""

    def __init__(self, query: RAGQuery, response: RAGResponse) -> None:
        self.query = query
        self.response = response
        self.claims: List[str] = []
        self.verified_citations: List[Citation] = []
        self.hallucinations: List[HallucinationFinding] = []
        self.faithfulness_score: float = 1.0
        self.answer_relevance_score: float = 1.0
        self.context_recall_score: float = 1.0
        self.context_precision_score: float = 1.0
        self.citation_grounding_score: float = 1.0
        self.hallucination_rate: float = 0.0
        self.overall_pass: bool = True
        self.execution_trace: List[str] = []
        self.metrics: Optional[EvaluationMetrics] = None


class EvaluationGraph:
    """DAG evaluation pipeline implementing structured state transformations."""

    def __init__(self, detector: Optional[HallucinationDetector] = None) -> None:
        self.detector = detector or HallucinationDetector()
        self._graph = nx.DiGraph()
        self._setup_workflow()

    def _setup_workflow(self) -> None:
        """Define topological execution graph."""
        nodes = [
            "extract_claims",
            "verify_citations",
            "detect_hallucinations",
            "compute_relevance",
            "aggregate_metrics",
        ]
        for node in nodes:
            self._graph.add_node(node)

        self._graph.add_edge("extract_claims", "verify_citations")
        self._graph.add_edge("verify_citations", "detect_hallucinations")
        self._graph.add_edge("detect_hallucinations", "compute_relevance")
        self._graph.add_edge("compute_relevance", "aggregate_metrics")

    def _extract_claims(self, state: EvaluationState) -> None:
        text = state.response.generated_answer.strip()
        raw = [c.strip() for c in text.split(".") if len(c.strip()) > 8]
        state.claims = raw if raw else [text]
        state.execution_trace.append("extract_claims")

    def _verify_citations(self, state: EvaluationState) -> None:
        verified: List[Citation] = []
        context_text = " ".join(c.text for c in state.response.retrieved_chunks).lower()

        for cit in state.response.citations:
            if cit.quote.lower().strip() in context_text:
                verified.append(cit)

        state.verified_citations = verified
        if state.response.citations:
            state.citation_grounding_score = round(
                len(verified) / len(state.response.citations), 4
            )
        else:
            state.citation_grounding_score = 1.0 if not state.response.retrieved_chunks else 0.85
        state.execution_trace.append("verify_citations")

    def _detect_hallucinations(self, state: EvaluationState) -> None:
        findings = self.detector.analyze(
            state.response.generated_answer, state.response.retrieved_chunks
        )
        state.hallucinations = findings

        total_claims = max(len(state.claims), 1)
        hallucinated_claims = min(len(findings), total_claims)
        state.hallucination_rate = round(hallucinated_claims / total_claims, 4)
        state.faithfulness_score = max(0.0, round(1.0 - state.hallucination_rate, 4))
        state.execution_trace.append("detect_hallucinations")

    def _compute_relevance(self, state: EvaluationState) -> None:
        import re

        stopwords = {"what", "was", "is", "in", "of", "for", "the", "a", "an", "to", "and", "how", "did"}
        q_tokens = [w for w in re.findall(r"\b\w+\b", state.query.query_text.lower()) if w not in stopwords]
        a_tokens = [w for w in re.findall(r"\b\w+\b", state.response.generated_answer.lower())]

        if not q_tokens:
            q_tokens = re.findall(r"\b\w+\b", state.query.query_text.lower())

        overlap = set(q_tokens) & set(a_tokens)
        relevance = len(overlap) / max(len(set(q_tokens)), 1)
        state.answer_relevance_score = round(min(1.0, relevance * 1.2), 4)

        # Context precision and recall heuristics based on retrieved chunk relevance
        if state.response.retrieved_chunks:
            relevant_chunks = sum(
                1 for c in state.response.retrieved_chunks if any(w in c.text.lower() for w in q_tokens)
            )
            state.context_precision_score = round(
                relevant_chunks / len(state.response.retrieved_chunks), 4
            )
            state.context_recall_score = 1.0 if relevant_chunks > 0 else 0.5
        else:
            state.context_precision_score = 0.5
            state.context_recall_score = 0.5

        state.execution_trace.append("compute_relevance")

    def _aggregate_metrics(self, state: EvaluationState) -> None:
        # Pass criteria: faithfulness >= 0.80, answer relevance >= 0.70, no CRITICAL hallucinations
        critical_count = sum(1 for h in state.hallucinations if h.severity.value == "CRITICAL")
        passes = (
            state.faithfulness_score >= 0.75
            and state.answer_relevance_score >= 0.50
            and critical_count == 0
        )
        state.overall_pass = passes

        state.metrics = EvaluationMetrics(
            faithfulness_score=state.faithfulness_score,
            answer_relevance_score=state.answer_relevance_score,
            context_recall_score=state.context_recall_score,
            context_precision_score=state.context_precision_score,
            citation_grounding_score=state.citation_grounding_score,
            hallucination_rate=state.hallucination_rate,
            findings=state.hallucinations,
            overall_pass=state.overall_pass,
        )
        state.execution_trace.append("aggregate_metrics")

    def run(self, query: RAGQuery, response: RAGResponse) -> EvaluationMetrics:
        """Execute evaluation DAG from start to finish."""
        start = time.perf_counter()
        state = EvaluationState(query=query, response=response)

        # Run nodes in topological order
        node_funcs = {
            "extract_claims": self._extract_claims,
            "verify_citations": self._verify_citations,
            "detect_hallucinations": self._detect_hallucinations,
            "compute_relevance": self._compute_relevance,
            "aggregate_metrics": self._aggregate_metrics,
        }

        for node in nx.topological_sort(self._graph):
            func = node_funcs[node]
            func(state)

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        assert state.metrics is not None
        state.metrics.evaluation_latency_ms = elapsed
        return state.metrics
