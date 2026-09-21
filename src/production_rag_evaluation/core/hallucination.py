"""Hallucination detection engine specialized for financial transcripts and 10-K filings."""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from production_rag_evaluation.models.schemas import (
    ClaimStatus,
    ClaimVerificationResult,
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
            "increased",
            "grew",
            "growth",
            "rose",
            "expanded",
            "up",
            "surpassed",
            "outperformed",
            "higher",
        }
        self._negative_directions = {
            "decreased",
            "fell",
            "dropped",
            "declined",
            "contracted",
            "down",
            "missed",
            "lower",
            "slumped",
        }

        # Ignored non-ticker capitalized tokens
        self._excluded_ticker_tokens: Set[str] = {
            "USD", "EUR", "GBP", "CAD", "JPY", "SEC", "CEO", "CFO", "CTO", "COO",
            "GAAP", "EPS", "FY", "Q1", "Q2", "Q3", "Q4", "AI", "IT", "US", "UK",
            "EU", "CAGR", "EBITDA", "ROIC", "IPO", "SP", "MDA", "LLC", "INC",
            "CORP", "LTD", "PLC", "ETF", "NAV", "AUM", "VAR", "CVAR", "ITEM",
            "NOTE", "FORM", "TOTAL", "NET", "THE", "AND", "FOR", "NOT", "ALL",
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
                        explanation=f"Generated value '{g_raw}' does not match any financial figure in retrieved context within {self.numerical_tolerance_pct * 100}% tolerance.",
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

    def detect_entity_swaps(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Detect entity swaps or foreign tickers present in answer but missing from context."""
        findings: List[HallucinationFinding] = []
        if not context_chunks:
            return findings

        context_tickers = {c.ticker.upper() for c in context_chunks if c.ticker}
        all_context_text = " ".join(c.text for c in context_chunks)
        context_words_upper = set(re.findall(r"\b[A-Z]{2,5}\b", all_context_text))

        # Look for uppercase tickers in answer
        gen_caps = set(re.findall(r"\b[A-Z]{2,5}\b", generated_answer))
        candidate_tickers = gen_caps - self._excluded_ticker_tokens

        for token in sorted(candidate_tickers):
            if token not in context_tickers and token not in context_words_upper:
                findings.append(
                    HallucinationFinding(
                        finding_type=HallucinationType.ENTITY_SWAP,
                        severity=Severity.HIGH,
                        claim_text=f"Referenced ticker/entity: {token}",
                        reference_text=f"Context tickers: {sorted(list(context_tickers))}",
                        confidence=0.85,
                        explanation=(
                            f"Answer mentions entity/ticker '{token}' which is not present in "
                            "retrieved context chunks."
                        ),
                    )
                )
        return findings

    def detect_temporal_mismatches(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Detect temporal mismatches between reporting years/quarters in answer vs context."""
        findings: List[HallucinationFinding] = []
        if not context_chunks:
            return findings

        context_full = " ".join(
            f"{c.text} {c.period or ''}" for c in context_chunks
        )
        context_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", context_full))
        if not context_years:
            return findings

        gen_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", generated_answer))
        mismatched_years = gen_years - context_years

        for yr in sorted(mismatched_years):
            findings.append(
                HallucinationFinding(
                    finding_type=HallucinationType.TEMPORAL_MISMATCH,
                    severity=Severity.MEDIUM,
                    claim_text=f"Reported temporal period/year: {yr}",
                    reference_text=f"Available context years: {sorted(list(context_years))}",
                    confidence=0.80,
                    explanation=(
                        f"Generated answer references period/year '{yr}', but retrieved context "
                        f"is restricted to {sorted(list(context_years))}."
                    ),
                )
            )
        return findings

    def verify_claims(
        self, claims: List[str], context_chunks: List[DocumentChunk]
    ) -> List[ClaimVerificationResult]:
        """Verify individual claim statements against context chunks and assign verification status."""
        results: List[ClaimVerificationResult] = []
        if not claims:
            return results

        for idx, claim in enumerate(claims, 1):
            claim_id = f"claim-{idx:03d}"
            claim_lower = claim.lower()
            claim_tokens = set(re.findall(r"\b\w{3,}\b", claim_lower))

            # Identify supporting chunks
            evidence_chunk_ids: List[str] = []
            max_chunk_overlap = 0.0
            for chunk in context_chunks:
                c_tokens = set(re.findall(r"\b\w{3,}\b", chunk.text.lower()))
                overlap = len(claim_tokens & c_tokens) / max(len(claim_tokens), 1)
                if overlap > 0.20:
                    evidence_chunk_ids.append(chunk.chunk_id)
                if overlap > max_chunk_overlap:
                    max_chunk_overlap = overlap

            # Contradiction check on single claim
            contra_findings = self.detect_directional_contradictions(claim, context_chunks)
            num_distortions = self.detect_numerical_distortions(claim, context_chunks)

            if contra_findings or num_distortions:
                status = ClaimStatus.REFUTED
                conf = 0.90
                reason = "Claim contradicts numerical facts or directional trends documented in context."
            elif not evidence_chunk_ids or not context_chunks:
                status = ClaimStatus.UNGROUNDED
                conf = 0.85
                reason = "Claim lacks supporting evidence chunks in retrieved context."
            elif max_chunk_overlap >= 0.35:
                status = ClaimStatus.VERIFIED
                conf = round(min(1.0, 0.70 + max_chunk_overlap * 0.3), 2)
                reason = f"Claim grounded in evidence chunks with {max_chunk_overlap:.1%} keyword overlap."
            elif max_chunk_overlap >= 0.20:
                status = ClaimStatus.AMBIGUOUS
                conf = 0.50
                reason = f"Claim partially evidenced ({max_chunk_overlap:.1%} overlap) but inconclusive."
            else:
                status = ClaimStatus.UNGROUNDED
                conf = 0.75
                reason = "Claim token overlap insufficient for grounding verification."

            results.append(
                ClaimVerificationResult(
                    claim_id=claim_id,
                    claim_text=claim,
                    status=status,
                    confidence=conf,
                    evidence_chunk_ids=evidence_chunk_ids,
                    explanation=reason,
                )
            )

        return results

    def analyze(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> List[HallucinationFinding]:
        """Run all hallucination detection rules and return aggregated findings."""
        findings: List[HallucinationFinding] = []
        findings.extend(self.detect_numerical_distortions(generated_answer, context_chunks))
        findings.extend(self.detect_directional_contradictions(generated_answer, context_chunks))
        findings.extend(self.detect_ungrounded_claims(generated_answer, context_chunks))
        findings.extend(self.detect_entity_swaps(generated_answer, context_chunks))
        findings.extend(self.detect_temporal_mismatches(generated_answer, context_chunks))
        return findings
