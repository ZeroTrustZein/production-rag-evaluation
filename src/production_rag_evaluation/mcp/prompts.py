"""Prompt templates and registry for MCP guided evaluation workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from production_rag_evaluation.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPPromptMessage,
)


class MCPPromptRegistry:
    """Registry managing standard prompt templates and message rendering."""

    def __init__(self) -> None:
        self._prompts: Dict[str, MCPPrompt] = {}

    def register(self, prompt: MCPPrompt) -> None:
        """Register a new MCP prompt template."""
        self._prompts[prompt.name] = prompt

    def get(self, name: str) -> Optional[MCPPrompt]:
        """Fetch prompt definition by name."""
        return self._prompts.get(name)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """Return prompts formatted per MCP prompts/list specification."""
        return [
            {
                "name": p.name,
                "description": p.description,
                "arguments": [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required,
                    }
                    for arg in p.arguments
                ],
            }
            for p in self._prompts.values()
        ]

    def render(self, name: str, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Render prompt arguments into MCP prompts/get messages response."""
        prompt = self.get(name)
        if prompt is None:
            raise KeyError(f"Prompt '{name}' not found in registry")

        # Validate mandatory arguments
        for arg in prompt.arguments:
            if arg.required and arg.name not in arguments:
                raise ValueError(
                    f"Missing required prompt argument '{arg.name}' for prompt '{name}'"
                )

        template = prompt.template or ""
        # Safe replacement of placeholders
        rendered_text = template
        for k, v in arguments.items():
            rendered_text = rendered_text.replace(f"{{{k}}}", str(v))

        msg = MCPPromptMessage(
            role="user",
            content={"type": "text", "text": rendered_text},
        )

        return {
            "description": prompt.description,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            ],
        }


def get_default_prompt_registry() -> MCPPromptRegistry:
    """Initialize and populate default prompt registry."""
    reg = MCPPromptRegistry()

    reg.register(
        MCPPrompt(
            name="earnings_call_audit",
            description="Audit quarterly financial earnings call transcript Q&A against model summary.",
            arguments=[
                MCPPromptArgument(
                    name="company", description="Company or firm name", required=True
                ),
                MCPPromptArgument(
                    name="quarter", description="Fiscal quarter (e.g. Q3 2024)", required=True
                ),
                MCPPromptArgument(
                    name="transcript_segment",
                    description="Raw transcript excerpt of CEO/CFO remarks",
                    required=True,
                ),
                MCPPromptArgument(
                    name="generated_summary",
                    description="Candidate summary or generated answer",
                    required=True,
                ),
            ],
            template="""You are a senior financial analyst and compliance auditor.
Audit the following generated summary against the official earnings call transcript for {company} ({quarter}).

TRANSCRIPT EXCERPT:
\"\"\"
{transcript_segment}
\"\"\"

GENERATED SUMMARY UNDER AUDIT:
\"\"\"
{generated_summary}
\"\"\"

Please evaluate:
1. Numerical Accuracy: Are all revenues, margins, EPS, and guidance numbers identical within 1%?
2. Directional Integrity: Does the summary misrepresent growth vs. contraction or risks?
3. Ungrounded Statements: Flag any claims not supported by the transcript excerpt.
Provide a clear verdict: PASS, CONDITIONAL_PASS, or FAIL with itemized citations.
""",
        )
    )

    reg.register(
        MCPPrompt(
            name="sec_10k_verification",
            description="Cross-reference specific claims against audited SEC Form 10-K disclosures.",
            arguments=[
                MCPPromptArgument(name="ticker", description="Equity ticker symbol", required=True),
                MCPPromptArgument(
                    name="fiscal_year",
                    description="Fiscal year period (e.g. FY2024)",
                    required=True,
                ),
                MCPPromptArgument(
                    name="filing_excerpt",
                    description="Audited 10-K section text (e.g. Item 7 MD&A or Note 12)",
                    required=True,
                ),
                MCPPromptArgument(
                    name="model_claim",
                    description="Specific numerical or legal claim made by RAG model",
                    required=True,
                ),
            ],
            template="""Cross-examine the candidate claim against the audited SEC Form 10-K filing for {ticker} ({fiscal_year}).

10-K FILING SECTION:
\"\"\"
{filing_excerpt}
\"\"\"

CANDIDATE MODEL CLAIM:
\"\"\"
{model_claim}
\"\"\"

Instructions:
1. Locate the exact sentence or financial table confirming or refuting the claim.
2. Note if the claim omits material risk factors, contingencies, or one-off impairment charges.
3. Classify finding severity: LOW, MEDIUM, HIGH, or CRITICAL.
""",
        )
    )

    reg.register(
        MCPPrompt(
            name="portfolio_risk_audit",
            description="Interpret quantitative risk metrics and stress test output for portfolio risk management.",
            arguments=[
                MCPPromptArgument(
                    name="portfolio_id", description="Portfolio code or label", required=True
                ),
                MCPPromptArgument(
                    name="risk_metrics_json",
                    description="JSON metrics containing Sharpe, Sortino, VaR, CVaR",
                    required=True,
                ),
                MCPPromptArgument(
                    name="stress_test_json",
                    description="JSON results of Monte Carlo stress scenario simulation",
                    required=True,
                ),
            ],
            template="""Conduct an institutional risk oversight evaluation for portfolio {portfolio_id}.

QUANTITATIVE RISK METRICS:
\"\"\"
{risk_metrics_json}
\"\"\"

MONTE CARLO STRESS TEST RESULTS:
\"\"\"
{stress_test_json}
\"\"\"

Tasks:
1. Assess tail risk adequacy: Compare 95% vs. 99% CVaR against simulated maximum drawdown.
2. Evaluate risk-adjusted return viability (Sharpe and Sortino relative to risk-free benchmark).
3. Recommend position hedging or capital allocation adjustments if stressed loss probability exceeds 15%.
""",
        )
    )

    return reg
