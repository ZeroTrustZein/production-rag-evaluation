"""CLI entry point for python -m production_rag_evaluation."""

import sys

from production_rag_evaluation.cli.main import cli

if __name__ == "__main__":
    sys.exit(cli())
