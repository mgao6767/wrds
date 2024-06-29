import warnings
import itertools
from typing import List
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, OptimizeResult

USE_CPP_EXTENSION = True
try:
    import frds.algorithms.utils.utils_ext as ext
except ImportError:
    USE_CPP_EXTENSION = False


class GARCHModel:
    """:doc:`/algorithms/garch` model with constant mean and Normal noise

    It estimates the model parameters only. No standard errors calculated.

    This code is heavily based on the `arch <https://arch.readthedocs.io/>`_.
    Modifications are made for easier understaing of the code flow.
    """

    @dataclass
    class Parameters:
        mu: float = np.nan
        omega: float = np.nan
        alpha: float = np.nan
        beta: float = np.nan
        loglikelihood: float = np.nan

    def __init__(self, returns: np.ndarray, zero_mean=False) -> None:
        """__init__

        Args:
            returns (np.ndarray): ``(T,)`` array of ``T`` returns
            zero_mean (bool): whether to use a zero mean returns model. Default to False.

        .. note:: ``returns`` is best to be percentage returns for optimization
        """
        self.returns = np.asarray(returns, dtype=np.float64)
        self.estimation_success = False
        self.loglikelihood_final = np.nan
        self.parameters = type(self).Parameters()
        self.resids = np.empty_like(self.returns)
        self.sigma2 = np.empty_like(self.returns)
        self.backcast_value = np.nan
        self.var_bounds: np.ndarray = None
        self.zero_mean = zero_mean

    def fit(self) -> Parameters:
        """Estimates the GARCH(1,1) parameters via MLE

        Returns:
            params: :class:`frds.algorithms.GARCHModel.Parameters`
        """
        # No repeated estimation?
        if self.estimation_success:
            return

        starting_vals = self.preparation()

        # Set bounds for parameters
        bounds = [
            (-np.inf, np.inf),  # No bounds for mu
            (1e-6, np.inf),  # Lower bound for omega
            (0.0, 1.0),  # Bounds for alpha
            (0.0, 1.0),  # Boudns for beta
        ]
        if self.zero_mean:
            bounds = bounds[1:]

        # Set constraint for stationarity
        def persistence_smaller_than_one(params: List[float]):
            alpha, beta = params[-2:]
            return 1.0 - (alpha + beta)

        # MLE via minimizing the negative log-likelihood
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
                args=(self.backcast_value, self.var_bounds, self.zero_mean),
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": persistence_smaller_than_one},
            )
            if opt.success:
                self.estimation_success = True
                if self.zero_mean:
                    self.parameters = type(self).Parameters(0.0, *list(opt.x), loglikelihood=-opt.fun)
                else:
                    self.parameters = type(self).Parameters(*list(opt.x), loglikelihood=-opt.fun)
                self.resids = self.returns - self.parameters.mu

        return self.parameters

    def preparation(self) -> List[float]:
        """Prepare starting values.

        Returns:
            List[float]: list of starting values
        """
        # Compute a starting value for volatility process by backcasting
        if self.zero_mean:
            resids = self.returns
        else:
            resids = self.returns - np.mean(self.returns)
        self.backcast_value = self.backcast(resids)
        # Compute a loose bound for the volatility process
        # This is to avoid NaN in MLE by avoiding zero/negative variance,
        # as well as unreasonably large variance.
        self.var_bounds = self.variance_bounds(resids)
        # Compute the starting values for MLE
        # Starting values for the volatility process
        var_params = self.starting_values(resids)
        # Starting value for mu is the sample mean return
        initial_mu = self.returns.mean()
        # Starting values are [mu, omega, alpha, beta]
        starting_vals = [initial_mu, *var_params]

        return starting_vals if not self.zero_mean else var_params

    def loglikelihood_model(
        self,
        params: np.ndarray,
        backcast: float,
        var_bounds: np.ndarray,
        zero_mean=False,
    ) -> float:
        """Calculates the negative log-likelihood based on the current ``params``.
        This function is used in optimization.

        Args:
            params (np.ndarray): [mu, omega, alpha, (gamma), beta]
            backcast (float): backcast value
            var_bounds (np.ndarray): variance bounds

        Returns:
            float: negative log-likelihood
        """
        if zero_mean:
            resids = self.returns
            self.sigma2 = self.compute_variance(params, resids, backcast, var_bounds)
        else:
            resids = self.returns - params[0]  # params[0] is mu
            self.sigma2 = self.compute_variance(
                params[1:], resids, backcast, var_bounds
            )
        return -self.loglikelihood(resids, self.sigma2)

    @staticmethod
    def loglikelihood(resids: np.ndarray, sigma2: np.ndarray) -> float:
        """Computes the log-likelihood assuming residuals are
        normally distributed conditional on the variance.

        Args:
            resids (np.ndarray): residuals to use in computing log-likelihood.
            sigma2 (np.ndarray): conditional variance of residuals.

        Returns:
            float: log-likelihood
        """
        l = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + resids**2.0 / sigma2)
        return np.sum(l)

    @staticmethod
    def compute_variance(
        params: List[float],
        resids: np.ndarray,
        backcast: float,
        var_bounds: np.ndarray,
    ) -> np.ndarray:
        """Computes the variances conditional on given parameters

        Args:
            params (List[float]): [omega, alpha, beta]
            resids (np.ndarray): residuals from mean equation
            backcast (float): backcast value
            var_bounds (np.ndarray): variance bounds

        Returns:
            np.ndarray: conditional variance
        """
        if USE_CPP_EXTENSION:
            return ext.compute_garch_variance(params, resids, backcast, var_bounds)
        omega, alpha, beta = params
        sigma2 = np.zeros_like(resids)
        sigma2[0] = omega + (alpha + beta) * backcast
        for t in range(1, len(resids)):
            sigma2[t] = omega + alpha * (resids[t - 1] ** 2) + beta * sigma2[t - 1]
            sigma2[t] = GARCHModel.bounds_check(sigma2[t], var_bounds[t])
        return sigma2

    def starting_values(self, resids: np.ndarray) -> List[float]:
        """Finds the optimal initial values for the volatility model via a grid
        search. For varying target persistence and alpha values, return the
        combination of alpha and beta that gives the highest loglikelihood.

        Args:
            resids (np.ndarray): residuals from the mean model

        Returns:
            List[float]: [omega, alpha, beta]
        """
        # Candidate target persistence - we don't have a prior knowledge here
        persistence_grid = [0.5, 0.7, 0.9, 0.98]
        # Candidate alpha values
        alpha_grid = [0.01, 0.05, 0.1, 0.2]
        # Sample variance
        var = self.returns.var()
        # Backcast initial value for the volaility process
        initial_params = []
        max_likelihood = -np.inf
        for alpha, p in itertools.product(*[alpha_grid, persistence_grid]):
            # Note that (alpha+beta) is the "persistence"
            beta = p - alpha
            # Note that the unconditional variance is omega/(1-persistence).
            # As we are guessing the initial value, we use the sample variance
            # as the unconditional variance to guess the omega value.
            omega = var * (1 - p)
            params = [omega, alpha, beta]
            sigma2 = self.compute_variance(
                params, resids, self.backcast_value, self.var_bounds
            )
            if -self.loglikelihood(resids, sigma2) > max_likelihood:
                initial_params = params
        return initial_params

    @staticmethod
    def variance_bounds(resids: np.ndarray) -> np.ndarray:
        """Compute bounds for conditional variances using EWMA.

        This function calculates the lower and upper bounds for conditional variances
        based on the residuals provided. The bounds are computed to ensure numerical
        stability during the parameter estimation process of GARCH models. The function
        uses Exponentially Weighted Moving Average (EWMA) to estimate the initial variance
        and then adjusts these estimates to lie within global bounds.

        Args:
            resids (np.ndarray): residuals from the mean model.

        Returns:
            np.ndarray: an array where each row contains the lower and upper bounds for the conditional variance at each time point.
        """

        T = len(resids)
        tau = min(75, T)
        # Compute initial variance using EWMA
        decay_factor = 0.94
        weights = decay_factor ** np.arange(tau)
        weights /= weights.sum()
        initial_variance = np.dot(weights, resids[:tau] ** 2)
        # Compute var_bound using EWMA (assuming ewma_recursion is defined)
        var_bound = GARCHModel.ewma(resids, initial_variance)
        # Compute global bounds
        global_lower_bound = resids.var() / 1e8
        global_upper_bound = 1e7 * (1 + (resids**2).max())
        # Adjust var_bound to ensure it lies within global bounds
        var_bound = np.clip(var_bound, global_lower_bound, global_upper_bound)
        # Create bounds matrix
        var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T

        return np.ascontiguousarray(var_bounds)

    @staticmethod
    def bounds_check(sigma2: float, var_bounds: np.ndarray) -> float:
        """Adjust the conditional variance at time t based on its bounds

        Args:
            sigma2 (float): conditional variance
            var_bounds (np.ndarray): lower and upper bounds

        Returns:
            float: adjusted conditional variance
        """
        if USE_CPP_EXTENSION:
            return ext.bounds_check(sigma2, var_bounds)
        lower, upper = var_bounds[0], var_bounds[1]
        sigma2 = max(lower, sigma2)
        if sigma2 > upper:
            if not np.isinf(sigma2):
                sigma2 = upper + np.log(sigma2 / upper)
            else:
                sigma2 = upper + 1000
        return sigma2

    @staticmethod
    def backcast(resids: np.ndarray) -> float:
        """Computes the starting value for estimating conditional variance.

        Args:
            resids (np.ndarray): residuals

        Returns:
            float: initial value from backcasting
        """
        # Limit to first tau observations to reduce computation
        tau = min(75, resids.shape[0])
        # Weights for Exponential Weighted Moving Average (EWMA)
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)  # Ensure weights add up to 1
        # Let the initial value to be the EWMA of first tau observations
        return float(np.sum((resids[:tau] ** 2) * w))

    @staticmethod
    def ewma(resids: np.ndarray, initial_value: float, lam=0.94) -> np.ndarray:
        """Compute the conditional variance estimates using
        Exponentially Weighted Moving Average (EWMA).

        Args:
            resids (np.ndarray): Residuals from the model.
            initial_value (float): Initial value for the conditional variance.
            lam (float): Decay factor for the EWMA.

        Returns:
            np.ndarray: Array containing the conditional variance estimates.
        """
        if USE_CPP_EXTENSION:
            return ext.ewma(resids, initial_value, lam)
        T = len(resids)
        variance = np.empty(T)
        variance[0] = initial_value  # Set the initial value
        # Compute the squared residuals
        squared_resids = resids**2
        # Compute the EWMA using the decay factors and squared residuals
        for t in range(1, T):
            variance[t] = lam * variance[t - 1] + (1 - lam) * squared_resids[t - 1]
        return variance


class GJRGARCHModel(GARCHModel):
    """:doc:`/algorithms/gjr-garch` model with constant mean and Normal noise

    It estimates the model parameters only. No standard errors calculated.

    This code is heavily based on the `arch <https://arch.readthedocs.io/>`_.
    Modifications are made for easier understaing of the code flow.
    """

    @dataclass
    class Parameters:
        mu: float = np.nan
        omega: float = np.nan
        alpha: float = np.nan
        gamma: float = np.nan
        beta: float = np.nan
        loglikelihood: float = np.nan

    def fit(self) -> Parameters:
        """Estimates the GJR-GARCH(1,1) parameters via MLE

        Returns:
            List[float]: [mu, omega, alpha, gamma, beta, loglikelihood]
        """
        # No repeated estimation?
        if self.estimation_success:
            return
        starting_vals = self.preparation()

        bounds = [
            (-np.inf, np.inf),  # No bounds for mu
            (1e-6, np.inf),  # Lower bound for omega
            (1e-6, 1.0),  # Bounds for alpha
            (1e-6, 1.0),  # Bounds for gamma
            (1e-6, 1.0),  # Boudns for beta
        ]
        if self.zero_mean:
            bounds = bounds[1:]

        # Set constraint for stationarity
        def persistence_smaller_than_one(params: List[float]):
            alpha, gamma, beta = params[-3:]
            return 1.0 - (alpha + beta + gamma / 2)

        # MLE via minimizing the negative log-likelihood
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step",
                RuntimeWarning,
            )
            opt: OptimizeResult = minimize(
                self.loglikelihood_model,
                starting_vals,
                args=(self.backcast_value, self.var_bounds, self.zero_mean),
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": persistence_smaller_than_one},
            )
            if opt.success:
                self.estimation_success = True
                if self.zero_mean:
                    self.parameters = type(self).Parameters(
                        0.0, *list(opt.x), loglikelihood=-opt.fun
                    )
                else:
                    self.parameters = type(self).Parameters(
                        *list(opt.x), loglikelihood=-opt.fun
                    )
                self.resids = self.returns - self.parameters.mu

        return self.parameters

    @staticmethod
    def compute_variance(
        params: List[float],
        resids: np.ndarray,
        backcast: float,
        var_bounds: np.ndarray,
    ) -> np.ndarray:
        """Computes the variances conditional on given parameters

        Args:
            params (List[float]): [omega, alpha, gamma, beta]
            resids (np.ndarray): residuals from mean equation
            backcast (float): backcast value
            var_bounds (np.ndarray): variance bounds

        Returns:
            np.ndarray: conditional variance
        """
        if USE_CPP_EXTENSION:
            return ext.compute_gjrgarch_variance(params, resids, backcast, var_bounds)
        # fmt: off
        omega, alpha, gamma, beta = params
        sigma2 = np.zeros_like(resids)
        sigma2[0] = omega + (alpha + gamma/2 + beta) * backcast
        for t in range(1, len(resids)):
            sigma2[t] = omega + alpha * (resids[t - 1] ** 2) + beta * sigma2[t - 1]
            sigma2[t] += gamma * resids[t - 1] ** 2 if resids[t - 1] < 0 else 0
            sigma2[t] = GJRGARCHModel.bounds_check(sigma2[t], var_bounds[t])
        return sigma2

    @staticmethod
    def forecast_variance(
        params: Parameters,
        resids: np.ndarray,
        initial_variance: float,
    ) -> np.ndarray:
        """Forecast the variances conditional on given parameters and residuals.

        Args:
            params (Parameters): :class:`frds.algorithms.GJRGARCHModel.Parameters`
            resids (np.ndarray): residuals to use
            initial_variance (float): starting value of variance forecasts

        Returns:
            np.ndarray: conditional variance
        """
        # fmt: off
        omega, alpha, gamma, beta = params.omega, params.alpha, params.gamma, params.beta
        sigma2 = np.zeros_like(resids)
        sigma2[0] = initial_variance
        for t in range(1, len(resids)):
            sigma2[t] = omega + alpha * (resids[t - 1] ** 2) + beta * sigma2[t - 1]
            sigma2[t] += gamma * resids[t - 1] ** 2 if resids[t - 1] < 0 else 0
        return sigma2[1:]

    def starting_values(self, resids: np.ndarray) -> List[float]:
        """Finds the optimal initial values for the volatility model via a grid
        search. For varying target persistence and alpha values, return the
        combination of alpha and beta that gives the highest loglikelihood.

        Args:
            resids (np.ndarray): residuals from the mean model

        Returns:
            List[float]: [omega, alpha, gamma, beta]
        """
        # Candidate target persistence - we don't have a prior knowledge here
        persistence_grid = [0.5, 0.7, 0.9, 0.98]
        # Candidate alpha values
        alpha_grid = [0.01, 0.05, 0.1, 0.2]
        gamma_grid = alpha_grid
        # Sample variance
        var = self.returns.var()
        # Backcast initial value for the volaility process
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)
        initial_params = []
        max_likelihood = -np.inf
        # fmt: off
        for alpha, gamma, p in itertools.product(*[alpha_grid, gamma_grid, persistence_grid]):
            # Note that (alpha+beta) is the "persistence"
            beta = p - alpha - gamma/2
            # Note that the unconditional variance is omega/(1-persistence).
            # As we are guessing the initial value, we use the sample variance
            # as the unconditional variance to guess the omega value.
            omega = var * (1 - p)
            params = [omega, alpha, gamma, beta]
            sigma2 = self.compute_variance(params, resids, backcast, var_bounds)
            if -self.loglikelihood(resids, sigma2) > max_likelihood:
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
    returns = df["nissan"].to_numpy() * 100

    g = GARCHModel(returns)
    res = g.fit()
    pprint(res)

    gjr = GJRGARCHModel(returns)
    res = gjr.fit()
    pprint(res)

    from arch import arch_model

    m = arch_model(returns, p=1, o=1, q=1)
    print(m.fit(disp=False))
