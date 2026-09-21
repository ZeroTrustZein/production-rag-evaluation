"""Hallucination detection engine specialized for financial transcripts and 10-K filings."""

from __future__ import annotations

import re
from typing import List, Tuple

from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    HallucinationFinding,
    HallucinationType,
    Severity,
)


class HallucinationDetector:
    """Detects factual contradictions, numerical distortions, and ungrounded statements."""

    def __init__(self, numerical_tolerance_pct: float = 0.01) -> None:
        self.numerical_tolerance_pct = numerical_tolerance_pct

        # Regex patterns for financial numbers ($12.5M, 15.4%, $120 billion, etc.)
        self._num_pattern = re.compile(
            r"(?:\$|€|£)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|billion|million|bn|m|k|trillion)?",
            re.IGNORECASE,
        )

        # Directional sentiment terms for contradiction checking
        self._positive_directions = {
            "increased", "grew", "growth", "rose", "expanded", "up", "surpassed", "outperformed", "higher"
        }
        self._negative_directions = {
            "decreased", "fell", "dropped", "declined", "contracted", "down", "missed", "lower", "slumped"
        }

    def _extract_numerical_values(self, text: str) -> List[Tuple[float, str, str]]:
        """Extract parsed float values with scale multiplier and full string."""
        results: List[Tuple[float, str, str]] = []
        for match in self._num_pattern.finditer(text):
            raw_num_str = match.group(1).replace(",", "")
            unit = (match.group(2) or "").lower()
            try:
                val = float(raw_num_str)
            except ValueError:
                continue

            multiplier = 1.0
            if unit in ("billion", "bn"):
                multiplier = 1e9
            elif unit in ("million", "m"):
                multiplier = 1e6
            elif unit == "k":
                multiplier = 1e3
            elif unit == "trillion":
                multiplier = 1e12

            normalized_val = val * multiplier
            results.append((normalized_val, unit, match.group(0).strip()))
        return results

    def _split_into_claims(self, text: str) -> List[str]:
        """Segment text into individual sentences / factual claim candidates."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        return claims

    def detect_numerical_distortions(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Identify numerical values in generated text that do not match context."""
        findings: List[HallucinationFinding] = []
        all_context_text = " ".join(c.text for c in context_chunks)
        context_numbers = self._extract_numerical_values(all_context_text)
        gen_numbers = self._extract_numerical_values(generated_answer)

        if not gen_numbers:
            return findings

        for g_val, g_unit, g_raw in gen_numbers:
            # Check if any context number is within tolerance
            matched = False
            for c_val, c_unit, c_raw in context_numbers:
                if c_val == 0 and g_val == 0:
                    matched = True
                    break
                denom = max(abs(c_val), 1e-6)
                rel_diff = abs(g_val - c_val) / denom
                if rel_diff <= self.numerical_tolerance_pct:
                    matched = True
                    break

            if not matched and abs(g_val) > 0.001:
                findings.append(
                    HallucinationFinding(
                        finding_type=HallucinationType.NUMERICAL_DISTORTION,
                        severity=Severity.HIGH,
                        claim_text=f"Reported value: {g_raw}",
                        reference_text=f"Available context numbers: {[c[2] for c in context_numbers[:5]]}",
                        confidence=0.92,
                        explanation=f"Generated value '{g_raw}' does not match any financial figure in retrieved context within {self.numerical_tolerance_pct*100}% tolerance.",
                    )
                )
        return findings

    def detect_directional_contradictions(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Detect directional contradiction (e.g. context says 'declined 4%', answer claims 'grew')."""
        findings: List[HallucinationFinding] = []
        all_context_lower = " ".join(c.text for c in context_chunks).lower()
        context_words = set(re.findall(r"\b\w+\b", all_context_lower))

        gen_claims = self._split_into_claims(generated_answer)
        for claim in gen_claims:
            claim_lower = claim.lower()
            claim_words = set(re.findall(r"\b\w+\b", claim_lower))

            has_pos = bool(claim_words & self._positive_directions)
            has_neg = bool(claim_words & self._negative_directions)

            ctx_has_pos = bool(context_words & self._positive_directions)
            ctx_has_neg = bool(context_words & self._negative_directions)

            if has_pos and not has_neg and ctx_has_neg and not ctx_has_pos:
                findings.append(
                    HallucinationFinding(
                        finding_type=HallucinationType.CONTRADICTION,
                        severity=Severity.CRITICAL,
                        claim_text=claim,
                        reference_text=all_context_lower[:250],
                        confidence=0.88,
                        explanation="Generated claim asserts growth/increase, but source context documents downward/declining movement.",
                    )
                )
            elif has_neg and not has_pos and ctx_has_pos and not ctx_has_neg:
                findings.append(
                    HallucinationFinding(
                        finding_type=HallucinationType.CONTRADICTION,
                        severity=Severity.CRITICAL,
                        claim_text=claim,
                        reference_text=all_context_lower[:250],
                        confidence=0.88,
                        explanation="Generated claim asserts decline/decrease, but source context documents growth/increase.",
                    )
                )

        return findings

    def detect_ungrounded_claims(
        self,
        generated_answer: str,
        context_chunks: List[DocumentChunk],
        min_overlap_ratio: float = 0.30,
    ) -> List[HallucinationFinding]:
        """Flag claims whose content words lack token grounding in context."""
        findings: List[HallucinationFinding] = []
        all_context_text = " ".join(c.text for c in context_chunks).lower()
        context_tokens = set(re.findall(r"\b\w{3,}\b", all_context_text))

        claims = self._split_into_claims(generated_answer)
        for claim in claims:
            claim_tokens = [w for w in re.findall(r"\b\w{3,}\b", claim.lower())]
            if not claim_tokens:
                continue

            grounded_tokens = [t for t in claim_tokens if t in context_tokens]
            overlap_ratio = len(grounded_tokens) / len(claim_tokens)

            if overlap_ratio < min_overlap_ratio:
                findings.append(
                    HallucinationFinding(
                        finding_type=HallucinationType.UNGROUNDED_CLAIM,
                        severity=Severity.MEDIUM,
                        claim_text=claim,
                        reference_text=all_context_text[:200],
                        confidence=round(1.0 - overlap_ratio, 2),
                        explanation=f"Claim has only {overlap_ratio:.1%} lexical overlap with retrieved context, suggesting unsupported extrapolation.",
                    )
                )

        return findings

    def analyze(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Run all hallucination detection rules and return aggregated findings."""
        findings: List[HallucinationFinding] = []
        findings.extend(self.detect_numerical_distortions(generated_answer, context_chunks))
        findings.extend(self.detect_directional_contradictions(generated_answer, context_chunks))
        findings.extend(self.detect_ungrounded_claims(generated_answer, context_chunks))
        return findings
