import warnings
from typing import List
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from frds.algorithms import GARCHModel


class GARCHModel_CCC:
    """:doc:`/algorithms/garch-ccc` model with the following specification:

    - Bivariate
    - Constant mean
    - Normal noise

    It estimates the model parameters only. No standard errors calculated.
    """

    @dataclass
    class Parameters:
        mu1: float = np.nan
        omega1: float = np.nan
        alpha1: float = np.nan
        beta1: float = np.nan
        mu2: float = np.nan
        omega2: float = np.nan
        alpha2: float = np.nan
        beta2: float = np.nan
        rho: float = np.nan
        loglikelihood: float = np.nan

    def __init__(self, returns1: np.ndarray, returns2: np.ndarray) -> None:
        """__init__

        Args:
            returns1 (np.ndarray): ``(T,)`` array of ``T`` returns of first asset
            returns2 (np.ndarray): ``(T,)`` array of ``T`` returns of second asset

        .. note:: ``returns`` is best to be percentage returns for optimization
        """
        self.returns1 = np.asarray(returns1, dtype=np.float64)
        self.returns2 = np.asarray(returns2, dtype=np.float64)
        self.model1 = GARCHModel(self.returns1)
        self.model2 = GARCHModel(self.returns2)
        self.estimation_success = False
        self.parameters = type(self).Parameters()

    def fit(self) -> Parameters:
        """Estimates the Multivariate GARCH(1,1) parameters via MLE

        Returns:
            params: :class:`frds.algorithms.GARCHModel_CCC.Parameters`
        """
        m1, m2 = self.model1, self.model2
        m1.fit()
        m2.fit()
        starting_vals = self.starting_values(m1.resids, m2.resids)

        # Step 4. Set bounds for parameters
        bounds = [
            # For first returns
            (None, None),  # No bounds for mu
            (1e-6, None),  # Lower bound for omega
            (0.0, 1.0),  # Bounds for alpha
            (0.0, 1.0),  # Boudns for beta
            # For second returns
            (None, None),  # No bounds for mu
            (1e-6, None),  # Lower bound for omega
            (0.0, 1.0),  # Bounds for alpha
            (0.0, 1.0),  # Boudns for beta
            # Constant correlation
            (-0.99, 0.99),  # Bounds for rho
        ]

        # Step 5. Set constraint for stationarity
        def persistence_smaller_than_one_1(params: List[float]):
            mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho = params
            return 1.0 - (alpha1 + beta1)

        def persistence_smaller_than_one_2(params: List[float]):
            mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho = params
            return 1.0 - (alpha2 + beta2)

        # fmt: off
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step",
                RuntimeWarning,
            )
            opt: OptimizeResult = minimize(
                self.loglikelihood_model,
                starting_vals,
                args=(m1.backcast_value, m2.backcast_value, m1.var_bounds, m2.var_bounds),
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": persistence_smaller_than_one_1},
                    {"type": "ineq", "fun": persistence_smaller_than_one_2},
                ],
            )
            if opt.success:
                self.estimation_success = True
                self.parameters = type(self).Parameters(*list(opt.x, ), loglikelihood=-opt.fun)
        return self.parameters

    def loglikelihood_model(
        self,
        params: np.ndarray,
        backcast1: float,
        backcast2: float,
        var_bounds1: np.ndarray,
        var_bounds2: np.ndarray,
    ) -> float:
        """Calculates the negative log-likelihood based on the current ``params``.

        Args:
            params (np.ndarray): [mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho]
            backcast1 (float): Backcast value for initializing the first return series variance.
            backcast2 (float): Backcast value for initializing the second return series variance.
            var_bounds1 (np.ndarray): Array of variance bounds for the first return series.
            var_bounds2 (np.ndarray): Array of variance bounds for the second return series.

        Returns:
            float: negative log-likelihood
        """
        # fmt: off
        mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho = params
        resids1 = self.returns1 - mu1
        resids2 = self.returns2 - mu2
        var_params1 = [omega1, alpha1, beta1]
        var_params2 = [omega2, alpha2, beta2]
        backcast1 = GARCHModel.backcast(resids1)
        backcast2 = GARCHModel.backcast(resids2)
        var_bounds1 = GARCHModel.variance_bounds(resids1)
        var_bounds2 = GARCHModel.variance_bounds(resids2)
        sigma2_1 = GARCHModel.compute_variance(var_params1, resids1, backcast1, var_bounds1)
        sigma2_2 = GARCHModel.compute_variance(var_params2, resids2, backcast2, var_bounds2)
        negative_loglikelihood = -self.loglikelihood(resids1, sigma2_1, resids2, sigma2_2, rho)
        return negative_loglikelihood

    def loglikelihood(
        self,
        resids1: np.ndarray,
        sigma2_1: np.ndarray,
        resids2: np.ndarray,
        sigma2_2: np.ndarray,
        rho: float,
    ) -> float:
        """
        Computes the log-likelihood for a bivariate GARCH(1,1) model with constant correlation.

        Args:
            resids1 (np.ndarray): Residuals for the first return series.
            sigma2_1 (np.ndarray): Array of conditional variances for the first return series.
            resids2 (np.ndarray): Residuals for the second return series.
            sigma2_2 (np.ndarray): Array of conditional variances for the second return series.
            rho (float): Constant correlation.

        Returns:
            float: The log-likelihood value for the bivariate model.

        """
        # z1 and z2 are standardized residuals
        z1 = resids1 / np.sqrt(sigma2_1)
        z2 = resids2 / np.sqrt(sigma2_2)
        # fmt: off
        log_likelihood_terms = -0.5 * (
            2 * np.log(2 * np.pi) 
            + np.log(sigma2_1 * sigma2_2 * (1 - rho ** 2))
            + (z1 ** 2  + z2 ** 2  - 2 * rho * z1 * z2) / (1 - rho ** 2)
        )
        log_likelihood = np.sum(log_likelihood_terms)
        return log_likelihood

    def starting_values(self, resids1: np.ndarray, resids2: np.ndarray) -> List[float]:
        """Finds the optimal initial values for the volatility model via a grid
        search. For varying target persistence and alpha values, return the
        combination of alpha and beta that gives the highest loglikelihood.

        Args:
            resids1 (np.ndarray): Array of residuals for the first return series.
            resids2 (np.ndarray): Array of residuals for the second return series.

        Returns:
            List[float]: [mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho]
        """
        m1, m2 = self.model1, self.model2
        # Constant correlation
        rho_grid = np.linspace(-0.9, 0.9, 10)

        initial_params = []
        max_likelihood = -np.inf
        for rho in rho_grid:
            # last one is loglikelihood
            m1_params = list(asdict(m1.parameters).values())[:-1]
            m2_params = list(asdict(m2.parameters).values())[:-1]
            params = [*m1_params, *m2_params, rho]
            ll = -self.loglikelihood_model(
                params,
                m1.backcast_value,
                m2.backcast_value,
                m1.var_bounds,
                m2.var_bounds,
            )
            if ll > max_likelihood:
                initial_params = params

        return initial_params


if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint

    df = pd.read_stata(
        "https://www.stata-press.com/data/r18/stocks.dta", convert_dates=["date"]
    )
    df.set_index("date", inplace=True)
    # Scale returns to percentage returns for better optimization results
    toyota = df["toyota"].to_numpy() * 100
    nissan = df["nissan"].to_numpy() * 100
    honda = df["honda"].to_numpy() * 100

    model = GARCHModel_CCC(toyota, honda)
    res = model.fit()
    pprint(res)
