"""Monte Carlo stress testing engine for financial risk and RAG hallucination sensitivity."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.stats as stats

from production_rag_evaluation.models.schemas import (
    MonteCarloStressResult,
    StressScenarioType,
    StressTestConfig,
)


class MonteCarloStressTester:
    """Simulates return distributions under adverse stress conditions and hallucinated inputs."""

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def simulate_gbm_paths(
        self,
        initial_value: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        horizon_days: int = 60,
        n_simulations: int = 1000,
        volatility_shock_multiplier: float = 1.0,
        drift_shock: float = 0.0,
    ) -> np.ndarray:
        """Simulate geometric brownian motion asset paths under optional volatility shock."""
        dt = 1.0 / 252.0
        eff_sigma = max(1e-6, sigma * volatility_shock_multiplier)
        eff_mu = mu + drift_shock

        # Generate standard normal shocks: shape (n_simulations, horizon_days)
        z = self._rng.standard_normal((n_simulations, horizon_days))
        daily_drift = (eff_mu - 0.5 * eff_sigma**2) * dt
        daily_diffusion = eff_sigma * np.sqrt(dt) * z

        daily_returns = np.exp(daily_drift + daily_diffusion)
        paths = np.empty((n_simulations, horizon_days + 1), dtype=np.float64)
        paths[:, 0] = initial_value
        paths[:, 1:] = initial_value * np.cumprod(daily_returns, axis=1)

        return paths

    def simulate_fat_tailed_paths(
        self,
        initial_value: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        horizon_days: int = 60,
        n_simulations: int = 1000,
        degrees_of_freedom: int = 4,
        volatility_shock_multiplier: float = 1.0,
        drift_shock: float = 0.0,
    ) -> np.ndarray:
        """Simulate fat-tailed paths using Student-t innovations to model severe black swan tail risk."""
        dt = 1.0 / 252.0
        eff_sigma = max(1e-6, sigma * volatility_shock_multiplier)
        eff_mu = mu + drift_shock

        # Student-t shocks with variance normalization: Var(t_df) = df / (df - 2)
        df = max(3, degrees_of_freedom)
        scale_factor = np.sqrt((df - 2.0) / df)
        z_raw = stats.t.rvs(df, size=(n_simulations, horizon_days), random_state=self.random_seed)
        z = z_raw * scale_factor

        daily_drift = (eff_mu - 0.5 * eff_sigma**2) * dt
        daily_diffusion = eff_sigma * np.sqrt(dt) * z

        daily_returns = np.exp(daily_drift + daily_diffusion)
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
        config: Optional[StressTestConfig] = None,
    ) -> MonteCarloStressResult:
        """Run stress simulation on empirical returns and assess resilience."""
        returns = np.asarray(returns, dtype=np.float64)
        valid_returns = returns[~np.isnan(returns)]

        # If config is provided, unpack parameters
        scenario = StressScenarioType.BASELINE
        drift_shock = 0.0
        init_val = 100.0

        if config is not None:
            n_simulations = config.n_simulations
            horizon_days = config.horizon_days
            volatility_shock = config.volatility_shock
            max_allowable_drawdown = config.max_allowable_drawdown
            scenario = config.scenario_type
            drift_shock = config.drift_shock
            init_val = config.initial_value
            if config.random_seed is not None:
                self.random_seed = config.random_seed
                self._rng = np.random.default_rng(config.random_seed)

        if len(valid_returns) < 2:
            mu = 0.05
            sigma = 0.15
        else:
            mu = float(np.mean(valid_returns)) * 252.0
            sigma = float(np.std(valid_returns, ddof=1)) * np.sqrt(252.0)

        if scenario == StressScenarioType.BLACK_SWAN:
            paths = self.simulate_fat_tailed_paths(
                initial_value=init_val,
                mu=mu,
                sigma=sigma,
                horizon_days=horizon_days,
                n_simulations=n_simulations,
                degrees_of_freedom=3,
                volatility_shock_multiplier=volatility_shock,
                drift_shock=drift_shock,
            )
        else:
            paths = self.simulate_gbm_paths(
                initial_value=init_val,
                mu=mu,
                sigma=sigma,
                horizon_days=horizon_days,
                n_simulations=n_simulations,
                volatility_shock_multiplier=volatility_shock,
                drift_shock=drift_shock,
            )

        terminal_values = paths[:, -1]
        p5 = float(np.percentile(terminal_values, 5))
        p50 = float(np.percentile(terminal_values, 50))
        p95 = float(np.percentile(terminal_values, 95))
        mean_term = float(np.mean(terminal_values))
        std_term = float(np.std(terminal_values, ddof=1))

        # Calculate max drawdowns per simulated trajectory
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdown_matrix = (running_max - paths) / np.maximum(running_max, 1e-6)
        max_drawdowns_per_sim = np.max(drawdown_matrix, axis=1)
        worst_drawdown = float(np.max(max_drawdowns_per_sim))

        # Breach probability: fraction of paths breaching max allowable drawdown
        breached_sims = np.sum(max_drawdowns_per_sim > max_allowable_drawdown)
        breach_prob = round(float(breached_sims / max(n_simulations, 1)), 4)

        stress_pass = worst_drawdown <= max_allowable_drawdown and p5 >= (init_val * 0.75)

        # Sample iteration summary data (first 25 terminal values)
        sample_iters = [round(float(v), 2) for v in terminal_values[:25]]

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
            scenario_type=scenario,
            breach_probability=breach_prob,
            iterations_data=sample_iters,
        )

    def run_scenario(
        self,
        returns: np.ndarray,
        scenario_type: StressScenarioType,
        config: Optional[StressTestConfig] = None,
    ) -> MonteCarloStressResult:
        """Execute prescribed macroeconomic and market stress scenario."""
        scenario_params = {
            StressScenarioType.BASELINE: {
                "volatility_shock": 1.0,
                "drift_shock": 0.0,
                "max_drawdown": 0.25,
            },
            StressScenarioType.VOLATILITY_SPIKE: {
                "volatility_shock": 2.5,
                "drift_shock": -0.05,
                "max_drawdown": 0.30,
            },
            StressScenarioType.LIQUIDITY_CRUNCH: {
                "volatility_shock": 3.0,
                "drift_shock": -0.15,
                "max_drawdown": 0.20,
            },
            StressScenarioType.RATE_SHOCK: {
                "volatility_shock": 1.8,
                "drift_shock": -0.10,
                "max_drawdown": 0.25,
            },
            StressScenarioType.BLACK_SWAN: {
                "volatility_shock": 3.5,
                "drift_shock": -0.30,
                "max_drawdown": 0.35,
            },
            StressScenarioType.HALLUCINATION_CASCADE: {
                "volatility_shock": 2.2,
                "drift_shock": -0.20,
                "max_drawdown": 0.25,
            },
        }

        params = scenario_params.get(scenario_type, scenario_params[StressScenarioType.BASELINE])

        cfg = config or StressTestConfig(
            scenario_type=scenario_type,
            volatility_shock=params["volatility_shock"],
            drift_shock=params["drift_shock"],
            max_allowable_drawdown=params["max_drawdown"],
        )
        cfg.scenario_type = scenario_type
        return self.run_stress_test(returns=returns, config=cfg)

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
                if self._rng.random() < hallucination_probability:
                    sign = 1.0 if self._rng.random() > 0.5 else -1.0
                    corrupted = val * (1.0 + sign * hallucination_distortion_pct)
                    simulated_estimates.append(corrupted)
                else:
                    simulated_estimates.append(val)

            sim_mean = float(np.mean(simulated_estimates))
            if abs(sim_mean - true_mean) / max(abs(true_mean), 1e-6) > 0.10:
                breaches += 1

        breach_rate = breaches / max(n_simulations, 1)
        return round(breach_rate, 4)
