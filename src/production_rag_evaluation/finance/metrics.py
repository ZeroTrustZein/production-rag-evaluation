"""Financial risk metrics engine: Sharpe, Sortino, Calmar, Max Drawdown, VaR, Beta, and Alpha."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.stats as stats

from production_rag_evaluation.models.schemas import RiskMetrics


class FinancialRiskEngine:
    """Computes production quantitative risk indicators on portfolio returns and trading series."""

    def __init__(self, risk_free_rate: float = 0.045, periods_per_year: int = 252) -> None:
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    @staticmethod
    def _clean_returns(returns: np.ndarray) -> np.ndarray:
        """Convert input series to 1D float64 array and filter out NaN entries."""
        arr = np.asarray(returns, dtype=np.float64)
        return arr[~np.isnan(arr)]

    def _get_rf_and_ppy(
        self,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> Tuple[float, int, float]:
        """Resolve effective annual risk-free rate, periods per year, and single-period Rf."""
        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        ppy = self.periods_per_year if periods_per_year is None else periods_per_year
        daily_rf = (1.0 + rf) ** (1.0 / ppy) - 1.0
        return rf, ppy, daily_rf

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate annualized Sharpe Ratio: (mean(R) - Rf_period) / std(R) * sqrt(periods)."""
        _rf, ppy, daily_rf = self._get_rf_and_ppy(risk_free_rate, periods_per_year)
        valid_returns = self._clean_returns(returns)
        if len(valid_returns) < 2:
            return 0.0

        excess_returns = valid_returns - daily_rf
        std = float(np.std(excess_returns, ddof=1))
        if std <= 1e-12 or np.isnan(std):
            return 0.0

        mean_excess = float(np.mean(excess_returns))
        sharpe = (mean_excess / std) * np.sqrt(ppy)
        return round(float(sharpe), 4)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate annualized Sortino Ratio focusing on downside volatility."""
        _rf, ppy, daily_rf = self._get_rf_and_ppy(risk_free_rate, periods_per_year)
        valid_returns = self._clean_returns(returns)
        if len(valid_returns) < 2:
            return 0.0

        excess_returns = valid_returns - daily_rf
        downside = valid_returns[valid_returns < target_return]
        if len(downside) == 0:
            downside_std = 1e-6
        else:
            downside_std = float(np.sqrt(np.mean(downside**2)))

        if downside_std <= 1e-12 or np.isnan(downside_std):
            return 0.0

        mean_excess = float(np.mean(excess_returns))
        sortino = (mean_excess / downside_std) * np.sqrt(ppy)
        return round(float(sortino), 4)

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum peak-to-trough decline (0.0 to 1.0)."""
        valid_returns = self._clean_returns(returns)
        if len(valid_returns) == 0:
            return 0.0

        wealth = np.insert(np.cumprod(1.0 + valid_returns), 0, 1.0)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = (running_max - wealth) / np.maximum(running_max, 1e-12)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        return round(max(0.0, min(1.0, max_dd)), 4)

    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Calculate Calmar Ratio: annualized return / max drawdown."""
        ppy = self.periods_per_year if periods_per_year is None else periods_per_year
        valid_returns = self._clean_returns(returns)
        if len(valid_returns) < 2:
            return 0.0

        mean_return = float(np.mean(valid_returns))
        ann_return = (1.0 + mean_return) ** ppy - 1.0
        max_dd = self.calculate_max_drawdown(valid_returns)

        if max_dd <= 1e-5:
            return round(ann_return * 10.0, 4)  # capped surrogate for zero drawdown
        return round(ann_return / max_dd, 4)

    def calculate_var_cvar(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute Historical Value-at-Risk and Conditional VaR (Expected Shortfall)."""
        valid_returns = self._clean_returns(returns)
        if len(valid_returns) == 0:
            return 0.0, 0.0

        alpha = (1.0 - confidence_level) * 100.0
        var = float(np.percentile(valid_returns, alpha))
        tail_losses = valid_returns[valid_returns <= var]
        cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var

        var_amount = round(abs(min(0.0, var)), 4)
        cvar_amount = round(abs(min(0.0, cvar)), 4)
        # Ensure CVaR is at least as large as VaR
        cvar_amount = max(var_amount, cvar_amount)
        return var_amount, cvar_amount

    def calculate_parametric_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        cornish_fisher: bool = True,
    ) -> float:
        """Compute Parametric VaR with optional Cornish-Fisher expansion for skew/kurtosis."""
        valid_returns = self._clean_returns(returns)
        if len(valid_returns) < 3:
            return 0.0

        mu = float(np.mean(valid_returns))
        sigma = float(np.std(valid_returns, ddof=1))
        if sigma <= 1e-12:
            return 0.0

        z = float(stats.norm.ppf(1.0 - confidence_level))

        if cornish_fisher and len(valid_returns) >= 8:
            s = float(stats.skew(valid_returns))
            k = float(stats.kurtosis(valid_returns))  # Fisher excess kurtosis
            z = (
                z
                + (z**2 - 1.0) * s / 6.0
                + (z**3 - 3.0 * z) * k / 24.0
                - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
            )

        var_return = mu + z * sigma
        return round(abs(min(0.0, var_return)), 4)

    def calculate_beta_alpha(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        periods_per_year: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute Beta and annualized Jensen's Alpha relative to a market benchmark."""
        _rf, ppy, daily_rf = self._get_rf_and_ppy(risk_free_rate, periods_per_year)

        ret = np.asarray(returns, dtype=np.float64)
        bench = np.asarray(benchmark_returns, dtype=np.float64)

        min_len = min(len(ret), len(bench))
        if min_len < 3:
            return 1.0, 0.0

        ret = ret[:min_len]
        bench = bench[:min_len]

        bench_var = float(np.var(bench, ddof=1))
        if bench_var <= 1e-12:
            return 1.0, 0.0

        covariance = float(np.cov(ret, bench, ddof=1)[0, 1])
        beta = covariance / bench_var

        ann_asset_excess = (float(np.mean(ret)) - daily_rf) * ppy
        ann_bench_excess = (float(np.mean(bench)) - daily_rf) * ppy
        alpha = ann_asset_excess - beta * ann_bench_excess

        return round(float(beta), 4), round(float(alpha), 4)

    def calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """Compute Information Ratio: annualized active return divided by tracking error."""
        ppy = self.periods_per_year if periods_per_year is None else periods_per_year
        ret = np.asarray(returns, dtype=np.float64)
        bench = np.asarray(benchmark_returns, dtype=np.float64)

        min_len = min(len(ret), len(bench))
        if min_len < 2:
            return 0.0

        active_returns = ret[:min_len] - bench[:min_len]
        tracking_error = float(np.std(active_returns, ddof=1))
        mean_active = float(np.mean(active_returns))
        if tracking_error <= 1e-12:
            return round(mean_active * np.sqrt(ppy) * 100.0, 4) if mean_active > 0 else 0.0

        ir = (mean_active / tracking_error) * np.sqrt(ppy)
        return round(float(ir), 4)

    def calculate_omega_ratio(
        self, returns: np.ndarray, threshold: float = 0.0
    ) -> float:
        """Compute Omega Ratio: probability-weighted ratio of gains vs losses."""
        valid = self._clean_returns(returns)
        if len(valid) == 0:
            return 1.0

        gains = np.sum(np.maximum(valid - threshold, 0.0))
        losses = np.sum(np.maximum(threshold - valid, 0.0))

        if losses <= 1e-12:
            return 100.0 if gains > 0 else 1.0

        omega = float(gains / losses)
        return round(omega, 4)

    def calculate_tail_risk(self, returns: np.ndarray) -> Tuple[float, float]:
        """Compute distribution skewness and excess kurtosis using scipy.stats."""
        valid = self._clean_returns(returns)
        if len(valid) < 3:
            return 0.0, 0.0

        skewness = float(stats.skew(valid))
        kurt = float(stats.kurtosis(valid))  # excess kurtosis
        return round(skewness, 4), round(kurt, 4)

    def compute_all_metrics(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[float] = None,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> RiskMetrics:
        """Compute comprehensive RiskMetrics suite from return array and optional benchmark."""
        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        ppy = self.periods_per_year

        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 2:
            returns = np.array([0.001, -0.0005, 0.002, 0.001], dtype=np.float64)

        sharpe = self.calculate_sharpe_ratio(returns, risk_free_rate=rf)
        sortino = self.calculate_sortino_ratio(returns, risk_free_rate=rf)
        max_dd = self.calculate_max_drawdown(returns)
        calmar = self.calculate_calmar_ratio(returns, risk_free_rate=rf)
        var_95, cvar_95 = self.calculate_var_cvar(returns, confidence_level=0.95)
        var_99, cvar_99 = self.calculate_var_cvar(returns, confidence_level=0.99)

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        ann_return = round(float((1.0 + mean_ret) ** ppy - 1.0), 4)
        ann_vol = round(float(std_ret * np.sqrt(ppy)), 4)

        downside = returns[returns < 0.0]
        downside_dev = (
            round(float(np.sqrt(np.mean(downside**2)) * np.sqrt(ppy)), 4)
            if len(downside) > 0
            else 0.0
        )

        beta: Optional[float] = None
        alpha: Optional[float] = None
        ir: Optional[float] = None

        if benchmark_returns is not None and len(benchmark_returns) >= 2:
            beta, alpha = self.calculate_beta_alpha(returns, benchmark_returns, risk_free_rate=rf)
            ir = self.calculate_information_ratio(returns, benchmark_returns)

        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            downside_deviation=downside_dev,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            information_ratio=ir,
            beta=beta,
            alpha=alpha,
        )
