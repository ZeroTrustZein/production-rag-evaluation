"""Stateful evaluation workflow graph using DAG state machine architecture."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from production_rag_evaluation.core.hallucination import HallucinationDetector
from production_rag_evaluation.models.schemas import (
    Citation,
    ClaimStatus,
    ClaimVerificationResult,
    EvaluationMetrics,
    EvaluationMode,
    HallucinationFinding,
    HallucinationType,
    RAGQuery,
    RAGResponse,
    Severity,
)

_RELEVANCE_STOPWORDS: frozenset[str] = frozenset(
    {
        "what",
        "was",
        "is",
        "in",
        "of",
        "for",
        "the",
        "a",
        "an",
        "to",
        "and",
        "how",
        "did",
    }
)
_WORD_PATTERN = re.compile(r"\b\w+\b")


class EvaluationState:
    """State object passing through evaluation graph nodes."""

    def __init__(
        self,
        query: RAGQuery,
        response: RAGResponse,
        mode: EvaluationMode = EvaluationMode.THOROUGH,
    ) -> None:
        self.query = query
        self.response = response
        self.mode = mode
        self.claims: List[str] = []
        self.verified_citations: List[Citation] = []
        self.hallucinations: List[HallucinationFinding] = []
        self.claim_verifications: List[ClaimVerificationResult] = []
        self.target_checks: Dict[str, Any] = {}
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
            "verify_claims",
            "verify_targets",
            "compute_relevance",
            "aggregate_metrics",
        ]
        for node in nodes:
            self._graph.add_node(node)

        self._graph.add_edge("extract_claims", "verify_citations")
        self._graph.add_edge("verify_citations", "detect_hallucinations")
        self._graph.add_edge("detect_hallucinations", "verify_claims")
        self._graph.add_edge("verify_claims", "verify_targets")
        self._graph.add_edge("verify_targets", "compute_relevance")
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
            state.citation_grounding_score = round(len(verified) / len(state.response.citations), 4)
        else:
            state.citation_grounding_score = 1.0 if not state.response.retrieved_chunks else 0.85
        state.execution_trace.append("verify_citations")

    def _detect_hallucinations(self, state: EvaluationState) -> None:
        findings = self.detector.analyze(
            state.response.generated_answer, state.response.retrieved_chunks
        )
        state.hallucinations = list(findings)

        total_claims = max(len(state.claims), 1)
        hallucinated_claims = min(len(findings), total_claims)
        state.hallucination_rate = round(hallucinated_claims / total_claims, 4)
        state.faithfulness_score = max(0.0, round(1.0 - state.hallucination_rate, 4))
        state.execution_trace.append("detect_hallucinations")

    def _verify_claims(self, state: EvaluationState) -> None:
        state.claim_verifications = self.detector.verify_claims(
            state.claims, state.response.retrieved_chunks
        )
        state.execution_trace.append("verify_claims")

    @staticmethod
    def _matches_target(
        target_val: float,
        extracted_nums: List[Tuple[float, str, str]],
        tolerance_pct: float,
    ) -> bool:
        """Check if target value matches any extracted number within percentage tolerance."""
        denom = max(abs(target_val), 1e-6)
        return any(
            abs(val - target_val) / denom <= tolerance_pct for val, _unit, _raw in extracted_nums
        )

    def _verify_targets(self, state: EvaluationState) -> None:
        target_results: Dict[str, Any] = {}
        answer_nums = self.detector._extract_numerical_values(state.response.generated_answer)

        # 1. Structured targets
        for target in state.query.structured_targets:
            matched = self._matches_target(target.target_value, answer_nums, target.tolerance_pct)
            target_results[target.metric_name] = {"target": target.target_value, "matched": matched}
            if not matched:
                state.hallucinations.append(
                    HallucinationFinding(
                        finding_type=HallucinationType.NUMERICAL_DISTORTION,
                        severity=Severity.HIGH,
                        claim_text=f"Expected {target.metric_name}: {target.target_value}",
                        reference_text=f"Tolerance: {target.tolerance_pct:.1%}",
                        confidence=0.90,
                        explanation=(
                            f"Structured target '{target.metric_name}' value {target.target_value} "
                            "missing or distorted in generated response."
                        ),
                    )
                )

        # 2. Raw expected numerical targets if structured_targets empty
        if not state.query.structured_targets and state.query.expected_numerical_targets:
            for exp_num in state.query.expected_numerical_targets:
                matched = self._matches_target(
                    exp_num, answer_nums, self.detector.numerical_tolerance_pct
                )
                target_results[str(exp_num)] = {"target": exp_num, "matched": matched}
                if not matched:
                    state.hallucinations.append(
                        HallucinationFinding(
                            finding_type=HallucinationType.NUMERICAL_DISTORTION,
                            severity=Severity.HIGH,
                            claim_text=f"Expected value: {exp_num}",
                            reference_text=f"Tolerance: {self.detector.numerical_tolerance_pct:.1%}",
                            confidence=0.88,
                            explanation=f"Numerical target {exp_num} missing or distorted in response.",
                        )
                    )

        state.target_checks = target_results
        state.execution_trace.append("verify_targets")

    def _compute_relevance(self, state: EvaluationState) -> None:
        q_text_lower = state.query.query_text.lower()
        q_tokens = [w for w in _WORD_PATTERN.findall(q_text_lower) if w not in _RELEVANCE_STOPWORDS]
        a_tokens = _WORD_PATTERN.findall(state.response.generated_answer.lower())

        if not q_tokens:
            q_tokens = _WORD_PATTERN.findall(q_text_lower)

        overlap = set(q_tokens) & set(a_tokens)
        relevance = len(overlap) / max(len(set(q_tokens)), 1)
        state.answer_relevance_score = round(min(1.0, relevance * 1.2), 4)

        # Context precision and recall heuristics based on retrieved chunk relevance
        if state.response.retrieved_chunks:
            relevant_chunks = sum(
                1
                for c in state.response.retrieved_chunks
                if any(w in c.text.lower() for w in q_tokens)
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
        critical_count = sum(1 for h in state.hallucinations if h.severity == Severity.CRITICAL)
        high_count = sum(1 for h in state.hallucinations if h.severity == Severity.HIGH)
        refuted_count = sum(
            1 for cv in state.claim_verifications if cv.status == ClaimStatus.REFUTED
        )

        if state.mode == EvaluationMode.FAST:
            passes = state.faithfulness_score >= 0.70 and critical_count == 0
        elif state.mode == EvaluationMode.FINANCIAL_STRICT:
            passes = (
                state.faithfulness_score >= 0.85
                and state.answer_relevance_score >= 0.60
                and critical_count == 0
                and high_count == 0
            )
        elif state.mode == EvaluationMode.AUDIT:
            passes = (
                state.faithfulness_score >= 0.75
                and state.answer_relevance_score >= 0.50
                and critical_count == 0
                and refuted_count == 0
            )
        else:  # THOROUGH
            passes = (
                state.faithfulness_score >= 0.75
                and state.answer_relevance_score >= 0.50
                and critical_count == 0
            )

        state.overall_pass = passes
        state.execution_trace.append("aggregate_metrics")

        state.metrics = EvaluationMetrics(
            faithfulness_score=state.faithfulness_score,
            answer_relevance_score=state.answer_relevance_score,
            context_recall_score=state.context_recall_score,
            context_precision_score=state.context_precision_score,
            citation_grounding_score=state.citation_grounding_score,
            hallucination_rate=state.hallucination_rate,
            findings=state.hallucinations,
            claim_verifications=state.claim_verifications,
            overall_pass=state.overall_pass,
            metadata={
                "evaluation_mode": state.mode.value,
                "execution_trace": list(state.execution_trace),
                "total_claims": len(state.claims),
                "target_checks": state.target_checks,
            },
        )

    def run(
        self,
        query: RAGQuery,
        response: RAGResponse,
        mode: EvaluationMode = EvaluationMode.THOROUGH,
    ) -> EvaluationMetrics:
        """Execute evaluation DAG from start to finish."""
        start = time.perf_counter()
        state = EvaluationState(query=query, response=response, mode=mode)

        # Run nodes in topological order
        node_funcs = {
            "extract_claims": self._extract_claims,
            "verify_citations": self._verify_citations,
            "detect_hallucinations": self._detect_hallucinations,
            "verify_claims": self._verify_claims,
            "verify_targets": self._verify_targets,
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
