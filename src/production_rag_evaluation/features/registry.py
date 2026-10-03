"""Centralized schema and feature definition registry."""

from __future__ import annotations

from typing import Dict, List, Optional

from production_rag_evaluation.features.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureRecord,
    FeatureSchema,
    FeatureValueType,
    SchemaValidationResult,
)


def _build_default_schemas() -> Dict[FeatureGroup, FeatureSchema]:
    """Construct production baseline schemas for all standard RAG feature groups."""
    return {
        FeatureGroup.QUERY: FeatureSchema(
            name="rag_query_features",
            feature_group=FeatureGroup.QUERY,
            version="1.0.0",
            features=[
                FeatureDefinition(
                    name="query_length",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.INT,
                    description="Total character length of the query",
                ),
                FeatureDefinition(
                    name="char_length",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.INT,
                    description="Alias for query character length",
                ),
                FeatureDefinition(
                    name="word_count",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.INT,
                    description="Total whitespace-delimited word count",
                ),
                FeatureDefinition(
                    name="has_question_mark",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Indicates interrogative punctuation",
                ),
                FeatureDefinition(
                    name="has_financial_keywords",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Presence of domain financial terms",
                ),
                FeatureDefinition(
                    name="has_ticker_mention",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Explicit equity ticker identification",
                ),
                FeatureDefinition(
                    name="structured_target_count",
                    feature_group=FeatureGroup.QUERY,
                    value_type=FeatureValueType.INT,
                    description="Number of structured quantitative targets",
                ),
            ],
        ),
        FeatureGroup.CHUNK: FeatureSchema(
            name="rag_chunk_features",
            feature_group=FeatureGroup.CHUNK,
            version="1.0.0",
            features=[
                FeatureDefinition(
                    name="chunk_length",
                    feature_group=FeatureGroup.CHUNK,
                    value_type=FeatureValueType.INT,
                    description="Context chunk character count",
                ),
                FeatureDefinition(
                    name="word_count",
                    feature_group=FeatureGroup.CHUNK,
                    value_type=FeatureValueType.INT,
                    description="Context chunk word count",
                ),
                FeatureDefinition(
                    name="has_sec_keywords",
                    feature_group=FeatureGroup.CHUNK,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Presence of regulatory SEC terminology",
                ),
                FeatureDefinition(
                    name="has_financial_metrics",
                    feature_group=FeatureGroup.CHUNK,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Contains balance sheet or income figures",
                ),
                FeatureDefinition(
                    name="has_numerical_data",
                    feature_group=FeatureGroup.CHUNK,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Presence of numeric tokens",
                ),
                FeatureDefinition(
                    name="period_specified",
                    feature_group=FeatureGroup.CHUNK,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Fiscal quarter or annual period identifier",
                ),
                FeatureDefinition(
                    name="ticker_present",
                    feature_group=FeatureGroup.CHUNK,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Document chunk associated ticker symbol",
                ),
            ],
        ),
        FeatureGroup.RESPONSE: FeatureSchema(
            name="rag_response_features",
            feature_group=FeatureGroup.RESPONSE,
            version="1.0.0",
            features=[
                FeatureDefinition(
                    name="answer_length",
                    feature_group=FeatureGroup.RESPONSE,
                    value_type=FeatureValueType.INT,
                    description="Length of generated answer",
                ),
                FeatureDefinition(
                    name="word_count",
                    feature_group=FeatureGroup.RESPONSE,
                    value_type=FeatureValueType.INT,
                    description="Word count of generated answer",
                ),
                FeatureDefinition(
                    name="citation_count",
                    feature_group=FeatureGroup.RESPONSE,
                    value_type=FeatureValueType.INT,
                    description="Number of cited source passages",
                ),
                FeatureDefinition(
                    name="retrieved_chunk_count",
                    feature_group=FeatureGroup.RESPONSE,
                    value_type=FeatureValueType.INT,
                    description="Total context chunks retrieved for generation",
                ),
                FeatureDefinition(
                    name="latency_ms",
                    feature_group=FeatureGroup.RESPONSE,
                    value_type=FeatureValueType.FLOAT,
                    description="Execution latency in milliseconds",
                ),
                FeatureDefinition(
                    name="has_currency_symbols",
                    feature_group=FeatureGroup.RESPONSE,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Presence of currency symbols ($/€/£)",
                ),
                FeatureDefinition(
                    name="has_percentages",
                    feature_group=FeatureGroup.RESPONSE,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Presence of percentage figures",
                ),
            ],
        ),
        FeatureGroup.EVALUATION: FeatureSchema(
            name="rag_evaluation_features",
            feature_group=FeatureGroup.EVALUATION,
            version="1.0.0",
            features=[
                FeatureDefinition(
                    name="faithfulness_score",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.FLOAT,
                    description="Faithfulness score between 0.0 and 1.0",
                ),
                FeatureDefinition(
                    name="answer_relevance_score",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.FLOAT,
                    description="Answer relevance score",
                ),
                FeatureDefinition(
                    name="context_recall_score",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.FLOAT,
                    description="Context recall score",
                ),
                FeatureDefinition(
                    name="context_precision_score",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.FLOAT,
                    description="Context precision score",
                ),
                FeatureDefinition(
                    name="citation_grounding_score",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.FLOAT,
                    description="Citation grounding precision",
                ),
                FeatureDefinition(
                    name="hallucination_rate",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.FLOAT,
                    description="Rate of detected hallucinated statements",
                ),
                FeatureDefinition(
                    name="overall_pass",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.BOOLEAN,
                    description="Binary evaluation pass/fail verdict",
                ),
                FeatureDefinition(
                    name="hallucination_count",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.INT,
                    description="Absolute count of hallucination findings",
                ),
                FeatureDefinition(
                    name="verified_claim_count",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.INT,
                    description="Count of grounded verified claims",
                ),
                FeatureDefinition(
                    name="refuted_claim_count",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.INT,
                    description="Count of refuted ungrounded claims",
                ),
                FeatureDefinition(
                    name="evaluation_latency_ms",
                    feature_group=FeatureGroup.EVALUATION,
                    value_type=FeatureValueType.FLOAT,
                    description="Latency of the evaluation graph pass",
                ),
            ],
        ),
        FeatureGroup.FINANCIAL: FeatureSchema(
            name="rag_financial_risk_features",
            feature_group=FeatureGroup.FINANCIAL,
            version="1.0.0",
            features=[
                FeatureDefinition(
                    name="sharpe_ratio",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Annualized Sharpe Ratio",
                ),
                FeatureDefinition(
                    name="sortino_ratio",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Annualized Sortino Ratio",
                ),
                FeatureDefinition(
                    name="max_drawdown",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Peak to trough maximum drawdown",
                ),
                FeatureDefinition(
                    name="calmar_ratio",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Calmar ratio (Return / Max Drawdown)",
                ),
                FeatureDefinition(
                    name="annualized_return",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Annualized expected asset return",
                ),
                FeatureDefinition(
                    name="annualized_volatility",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Annualized return standard deviation",
                ),
                FeatureDefinition(
                    name="downside_deviation",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Standard deviation of negative returns",
                ),
                FeatureDefinition(
                    name="var_95",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="1-day 95% Value at Risk",
                ),
                FeatureDefinition(
                    name="cvar_95",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="1-day 95% Conditional VaR / Expected Shortfall",
                ),
                FeatureDefinition(
                    name="var_99",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="1-day 99% Value at Risk",
                ),
                FeatureDefinition(
                    name="cvar_99",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="1-day 99% Conditional VaR",
                ),
                FeatureDefinition(
                    name="information_ratio",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Information ratio relative to benchmark",
                ),
                FeatureDefinition(
                    name="beta",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Beta sensitivity relative to benchmark",
                ),
                FeatureDefinition(
                    name="alpha",
                    feature_group=FeatureGroup.FINANCIAL,
                    value_type=FeatureValueType.FLOAT,
                    description="Alpha excess return",
                ),
            ],
        ),
    }


class FeatureRegistry:
    """Catalog managing schema specifications, type validations, and feature evolution."""

    def __init__(self, load_defaults: bool = True) -> None:
        self._schemas: Dict[FeatureGroup, FeatureSchema] = {}
        if load_defaults:
            self._schemas.update(_build_default_schemas())

    def register_schema(self, schema: FeatureSchema) -> None:
        """Register or overwrite a schema for a feature group."""
        self._schemas[schema.feature_group] = schema

    def get_schema(self, group: FeatureGroup) -> Optional[FeatureSchema]:
        """Fetch the registered schema for a feature group."""
        return self._schemas.get(group)

    def list_schemas(self) -> Dict[FeatureGroup, FeatureSchema]:
        """Retrieve all currently registered group schemas."""
        return dict(self._schemas)

    def get_feature_definition(
        self, group: FeatureGroup, feature_name: str
    ) -> Optional[FeatureDefinition]:
        """Look up a feature definition by group and feature name."""
        schema = self.get_schema(group)
        if schema is None:
            return None
        return schema.get_feature(feature_name)

    def validate_record(
        self, record: FeatureRecord, strict: bool = False
    ) -> SchemaValidationResult:
        """Validate feature values, types, and nullability against the registered schema."""
        schema = self.get_schema(record.feature_group)
        errors: List[str] = []
        warnings: List[str] = []

        if schema is None:
            if strict:
                errors.append(f"No registered schema found for group: {record.feature_group.value}")
            else:
                warnings.append(
                    f"No schema registered for group: {record.feature_group.value}; skipped validation."
                )
            return SchemaValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

        # Check all features in the record
        for fname, fval in record.features.items():
            fdef = schema.get_feature(fname)
            if fdef is None:
                if strict:
                    errors.append(
                        f"Unrecognized feature '{fname}' in group {record.feature_group.value}"
                    )
                else:
                    warnings.append(f"Feature '{fname}' is not defined in schema {schema.name}")
                continue

            # Null check
            if fval is None:
                if not fdef.nullable:
                    errors.append(f"Non-nullable feature '{fname}' received null value")
                continue

            # Type check
            if not fdef.value_type.is_valid(fval):
                errors.append(
                    f"Feature '{fname}' expected {fdef.value_type.value}, received {type(fval).__name__}"
                )

        # Check required non-nullable fields missing from record if strict
        if strict:
            for fdef in schema.features:
                if not fdef.nullable and fdef.name not in record.features:
                    errors.append(
                        f"Required non-nullable feature '{fdef.name}' missing from record"
                    )

        return SchemaValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


_DEFAULT_REGISTRY: Optional[FeatureRegistry] = None


def get_default_registry() -> FeatureRegistry:
    """Return globally shared singleton instance of the default FeatureRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = FeatureRegistry(load_defaults=True)
    return _DEFAULT_REGISTRY
