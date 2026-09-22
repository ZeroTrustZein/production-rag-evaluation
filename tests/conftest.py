"""Global fixtures and test configuration."""

import numpy as np
import pytest
from click.testing import CliRunner

from production_rag_evaluation.models.schemas import (
    DocumentChunk,
    DocumentType,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_chunk_apple() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-aapl-10k-01",
        text="For fiscal year 2024, Apple reported net sales of $391.0 billion, up 2% compared to 2023. Gross margin was 46.2%. Services revenue reached an all-time record of $96.2 billion.",
        doc_type=DocumentType.SEC_10K,
        ticker="AAPL",
        period="FY2024",
        section="Item 7. Management's Discussion and Analysis",
    )


@pytest.fixture
def sample_returns() -> np.ndarray:
    np.random.seed(42)
    # Generate 100 days of synthetic trading returns with mean 0.08/252, vol 0.18/sqrt(252)
    daily_mu = 0.08 / 252.0
    daily_sigma = 0.18 / np.sqrt(252.0)
    return np.random.normal(daily_mu, daily_sigma, 100)
