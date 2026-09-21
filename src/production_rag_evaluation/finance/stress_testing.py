"""Monte Carlo stress testing engine for financial risk and RAG hallucination sensitivity."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from production_rag_evaluation.models.schemas import MonteCarloStressResult


class MonteCarloStressTester:
    """Simulates return distributions under adverse stress conditions and hallucinated inputs."""

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_gbm_paths(
        self,
        initial_value: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        horizon_days: int = 60,
        n_simulations: int = 1000,
        volatility_shock_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Simulate geometric brownian motion asset paths under optional volatility shock."""
        dt = 1.0 / 252.0
        eff_sigma = sigma * volatility_shock_multiplier

        # Generate standard normal shocks: shape (n_simulations, horizon_days)
        z = np.random.standard_normal((n_simulations, horizon_days))
        daily_drift = (mu - 0.5 * eff_sigma**2) * dt
        daily_diffusion = eff_sigma * np.sqrt(dt) * z

        daily_returns = np.exp(daily_drift + daily_diffusion)
        # Cumulative product along horizon
        paths = np.empty((n_simulations, horizon_days + 1), dtype=np.float64)
        paths[:, 0] = initial_value
        paths[:, 1:] = initial_value * np.cumprod(daily_returns, axis=1)

        return paths

    def run_stress_test(
        self,
        returns: np.ndarray,
        n_simulations: int = 1000,
        horizon_days: int = 30,
        volatility_shock: float = 1.5,
        max_allowable_drawdown: float = 0.25,
    ) -> MonteCarloStressResult:
        """Run stress simulation on empirical returns and assess resilience."""
        if len(returns) < 2:
            mu = 0.05
            sigma = 0.15
        else:
            mu = float(np.mean(returns)) * 252.0
            sigma = float(np.std(returns, ddof=1)) * np.sqrt(252.0)

        paths = self.simulate_gbm_paths(
            initial_value=100.0,
            mu=mu,
            sigma=sigma,
            horizon_days=horizon_days,
            n_simulations=n_simulations,
            volatility_shock_multiplier=volatility_shock,
        )

        terminal_values = paths[:, -1]
        p5 = float(np.percentile(terminal_values, 5))
        p50 = float(np.percentile(terminal_values, 50))
        p95 = float(np.percentile(terminal_values, 95))
        mean_term = float(np.mean(terminal_values))
        std_term = float(np.std(terminal_values, ddof=1))

        # Calculate max drawdowns per simulated trajectory
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdown_matrix = (running_max - paths) / running_max
        max_drawdowns_per_sim = np.max(drawdown_matrix, axis=1)
        worst_drawdown = float(np.max(max_drawdowns_per_sim))

        stress_pass = worst_drawdown <= max_allowable_drawdown and p5 >= 75.0

        return MonteCarloStressResult(
            n_simulations=n_simulations,
            horizon_steps=horizon_days,
            p5_terminal=round(p5, 2),
            p50_terminal=round(p50, 2),
            p95_terminal=round(p95, 2),
            mean_terminal=round(mean_term, 2),
            std_terminal=round(std_term, 2),
            worst_case_drawdown=round(worst_drawdown, 4),
            stress_pass=stress_pass,
        )

    def stress_test_hallucination_impact(
        self,
        base_financial_estimates: List[float],
        hallucination_probability: float = 0.15,
        hallucination_distortion_pct: float = 0.25,
        n_simulations: int = 500,
    ) -> float:
        """Simulate how often hallucinated outputs trigger a >10% portfolio forecasting error."""
        if not base_financial_estimates:
            base_financial_estimates = [100.0, 105.0, 98.0, 112.0]

        true_mean = float(np.mean(base_financial_estimates))
        breaches = 0

        for _ in range(n_simulations):
            simulated_estimates = []
            for val in base_financial_estimates:
                if np.random.random() < hallucination_probability:
                    # Inject distortion in either direction
                    sign = 1.0 if np.random.random() > 0.5 else -1.0
                    corrupted = val * (1.0 + sign * hallucination_distortion_pct)
                    simulated_estimates.append(corrupted)
                else:
                    simulated_estimates.append(val)

            sim_mean = float(np.mean(simulated_estimates))
            if abs(sim_mean - true_mean) / max(abs(true_mean), 1e-6) > 0.10:
                breaches += 1

        breach_rate = breaches / n_simulations
        return round(breach_rate, 4)
