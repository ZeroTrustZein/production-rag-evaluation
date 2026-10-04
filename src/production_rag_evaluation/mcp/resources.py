"""Resource definitions and registry for MCP client reading."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from production_rag_evaluation import __version__
from production_rag_evaluation.features.registry import get_default_registry
from production_rag_evaluation.mcp.models import MCPResource
from production_rag_evaluation.models.schemas import StressScenarioType


class MCPResourceRegistry:
    """Registry maintaining readable resources and resolving URI lookups."""

    def __init__(self) -> None:
        self._resources: Dict[str, MCPResource] = {}
        self._providers: Dict[str, Callable[[], str]] = {}

    def register(
        self,
        resource: MCPResource,
        provider: Optional[Callable[[], str]] = None,
    ) -> None:
        """Register a resource representation and optional dynamic generator."""
        self._resources[resource.uri] = resource
        if provider is not None:
            self._providers[resource.uri] = provider

    def get(self, uri: str) -> Optional[MCPResource]:
        """Fetch static resource record by URI."""
        return self._resources.get(uri)

    def list_resources(self) -> List[Dict[str, Any]]:
        """Return resources formatted per MCP resources/list specification."""
        return [
            {
                "uri": res.uri,
                "name": res.name,
                "description": res.description,
                "mimeType": res.mimeType,
            }
            for res in self._resources.values()
        ]

    def read(self, uri: str) -> Dict[str, Any]:
        """Resolve resource payload and return MCP resources/read contents structure."""
        resource = self._resources.get(uri)
        if resource is None:
            raise KeyError(f"Resource with URI '{uri}' not found")

        provider = self._providers.get(uri)
        text_content = provider() if provider else (resource.text or "")

        return {
            "contents": [
                {
                    "uri": resource.uri,
                    "mimeType": resource.mimeType,
                    "text": text_content,
                }
            ]
        }


def _provide_health_status() -> str:
    """Dynamic generator for system health check resource."""
    payload = {
        "status": "HEALTHY",
        "version": __version__,
        "subsystems": {
            "rag_evaluator": "AVAILABLE",
            "semantic_cache": "AVAILABLE",
            "hallucination_detector": "AVAILABLE",
            "risk_engine": "AVAILABLE",
            "stress_tester": "AVAILABLE",
            "feature_store": "AVAILABLE",
        },
    }
    return json.dumps(payload, indent=2)


def _provide_features_schema() -> str:
    """Dynamic generator for feature catalog schema resource."""
    registry = get_default_registry()
    catalog: Dict[str, List[Dict[str, Any]]] = {}
    for d in registry.list_definitions():
        group_key = d.feature_group.value
        catalog.setdefault(group_key, []).append(
            {
                "name": d.name,
                "value_type": d.value_type.value,
                "description": d.description,
                "nullable": d.nullable,
            }
        )
    return json.dumps({"feature_groups": catalog}, indent=2)


def _provide_finance_scenarios() -> str:
    """Dynamic generator for available stress test scenarios."""
    scenarios = [
        {
            "name": s.value,
            "description": f"Simulation profile for {s.value.replace('_', ' ').title()}",
        }
        for s in StressScenarioType
    ]
    return json.dumps({"scenarios": scenarios}, indent=2)


def _provide_evaluations_summary() -> str:
    """Dynamic generator for evaluation thresholds and configuration."""
    summary = {
        "default_thresholds": {
            "faithfulness": 0.85,
            "answer_relevance": 0.80,
            "context_precision": 0.75,
            "claim_verification": 0.80,
            "numerical_tolerance_pct": 0.01,
        },
        "evaluation_modes": ["FAST", "BALANCED", "THOROUGH", "STRICT"],
    }
    return json.dumps(summary, indent=2)


def get_default_resource_registry() -> MCPResourceRegistry:
    """Instantiate and populate default MCP resource registry."""
    reg = MCPResourceRegistry()

    reg.register(
        MCPResource(
            uri="rag://system/health",
            name="System Health & Subsystem Status",
            description="Operational status and version diagnostics of all evaluation engines.",
            mimeType="application/json",
        ),
        provider=_provide_health_status,
    )

    reg.register(
        MCPResource(
            uri="rag://features/schema",
            name="Feature Store Catalog Schema",
            description="Schema definitions and registered feature groups in the streaming pipeline.",
            mimeType="application/json",
        ),
        provider=_provide_features_schema,
    )

    reg.register(
        MCPResource(
            uri="rag://finance/scenarios",
            name="Financial Stress Testing Scenarios",
            description="Enumeration of supported macro and micro shock testing profiles.",
            mimeType="application/json",
        ),
        provider=_provide_finance_scenarios,
    )

    reg.register(
        MCPResource(
            uri="rag://evaluations/summary",
            name="Evaluation Thresholds & Policies",
            description="Production pass/fail threshold targets and evaluation mode specifications.",
            mimeType="application/json",
        ),
        provider=_provide_evaluations_summary,
    )

    return reg
