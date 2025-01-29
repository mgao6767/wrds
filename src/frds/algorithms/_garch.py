import warnings
import itertools
from typing import List, Union
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
            self.sigma2 = compute_variance(params, resids, backcast, var_bounds)
        else:
            resids = self.returns - params[0]  # params[0] is mu
            self.sigma2 = compute_variance(
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
        if len(sigma2.shape)>1:
            return np.sum(l, axis=1)
        else:
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
        def hierarchical_grid_search(resids, initial_range, base_grid_size=10, depth=2):
            current_range = initial_range
            best_params = None

            for i in range(depth):
                if i != 0:
                    # 更新搜索范围为最佳参数周围的较小区域
                    search_width = (current_range[:, 1] - current_range[:, 0]) / base_grid_size
                    current_range = np.array([
                        [best_params[1] - search_width[0]*2, best_params[1] + search_width[0]*2],
                        [best_params[2] - search_width[1]*2, best_params[2] + search_width[1]*2]
                    ])
                # 创建当前深度的网格
                alpha_grid = np.linspace(current_range[0, 0], current_range[0, 1], base_grid_size)
                p_grid = np.linspace(current_range[1, 0], current_range[1, 1], base_grid_size)
                alpha_values, p_values = np.meshgrid(alpha_grid, p_grid)
                parameters = np.stack([alpha_values.ravel(), p_values.ravel()], axis=1)
                
                # 过滤无效组合
                valid_indices = np.logical_and(parameters[:, 1] - parameters[:, 0] > 0, # beta > 0
                                               parameters[:, 0] > 0) # alpha > 0
                valid_parameters = parameters[valid_indices]

                # 得到三个参数组成的二维矩阵
                alpha = valid_parameters[:, 0]
                p = valid_parameters[:, 1]
                beta = p - alpha
                omega = self.returns.var() * (1 - p)

                omega = omega.reshape(-1, 1)
                alpha = alpha.reshape(-1, 1)
                beta = beta.reshape(-1, 1)

                # Combine them into a new matrix
                valid_parameters = np.hstack((omega, alpha, beta))
                sigma2 = compute_variance(
                    valid_parameters, resids, self.backcast_value, self.var_bounds
                )
                ll_values = self.loglikelihood(resids, sigma2)
                max_ll_index = np.argmax(ll_values)
                best_params = valid_parameters[max_ll_index]
                
            return best_params

        # 初始化参数范围
        initial_range = np.array([[0.01, 0.2], [0.5, 0.99]])  # [a_range, b_range]
        base_grid_size = 5  # 初始网格大小
        depth = 1  # 搜索深度，即细化的次数

        # 调用函数
        best_parameters = hierarchical_grid_search(resids, initial_range, base_grid_size, depth)
        initial_params = best_parameters
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
        def hierarchical_grid_search(resids, initial_range, base_grid_size=10, depth=2):
            current_range = initial_range
            best_params = None

            for i in range(depth):
                if i != 0:
                    # 更新搜索范围为最佳参数周围的较小区域
                    search_width = (current_range[:, 1] - current_range[:, 0]) / base_grid_size
                    current_range = np.array([
                        [best_params[1] - search_width[0]*2, best_params[1] + search_width[0]*2],
                        [best_params[2] - search_width[1]*2, best_params[2] + search_width[1]*2],
                        [best_params[3] - search_width[2]*2, best_params[3] + search_width[2]*2]
                    ])
                # 创建当前深度的网格
                alpha_grid = np.linspace(current_range[0, 0], current_range[0, 1], base_grid_size)
                gamma_grid = np.linspace(current_range[1, 0], current_range[1, 1], base_grid_size)
                p_grid = np.linspace(current_range[2, 0], current_range[2, 1], base_grid_size)
                alpha_values, gamma_values, p_values = np.meshgrid(alpha_grid, gamma_grid, p_grid)
                parameters = np.stack([alpha_values.ravel(), gamma_values.ravel(), p_values.ravel()], axis=1)
                # 过滤无效组合
                valid_indices = np.logical_and(parameters[:, 2] - parameters[:, 0] - parameters[:, 1]/2 > 0, # beta > 0
                                               parameters[:, 1] > 0, # gamma>0
                                               parameters[:, 0] > 0) # alpha > 0
                valid_parameters = parameters[valid_indices]

                # 得到三个参数组成的二维矩阵
                alpha = valid_parameters[:, 0]
                gamma = valid_parameters[:, 1]
                p = valid_parameters[:, 2]
                beta = p - alpha - gamma/2
                omega = self.returns.var() * (1 - p)

                omega = omega.reshape(-1, 1)
                alpha = alpha.reshape(-1, 1)
                gamma = gamma.reshape(-1, 1)
                beta = beta.reshape(-1, 1)

                # Combine them into a new matrix
                valid_parameters = np.hstack((omega, alpha, gamma, beta))
                sigma2 = compute_variance(
                    valid_parameters, resids, self.backcast_value, self.var_bounds
                )
                ll_values = self.loglikelihood(resids, sigma2)
                max_ll_index = np.argmax(ll_values)
                best_params = valid_parameters[max_ll_index]
                
            return best_params

        # 初始化参数范围
        initial_range = np.array([[0.01, 0.2], [0.01, 0.2], [0.5, 0.99]])  # [a_range, b_range]
        base_grid_size = 5  # 初始网格大小
        depth = 1  # 搜索深度，即细化的次数

        # 调用函数
        best_parameters = hierarchical_grid_search(resids, initial_range, base_grid_size, depth)
        initial_params = best_parameters
        return initial_params

from numba import jit
@jit(nopython=True, cache=True)
def bounds_check(sigma2: np.float64|np.ndarray, var_bounds:np.ndarray) -> np.float64|np.ndarray:
    lower, upper = var_bounds[0], var_bounds[1]
    if isinstance(sigma2, np.float64):
        if sigma2 < lower:
            sigma2 = lower
        if sigma2 > upper:
            if not np.isinf(sigma2):
                sigma2 = upper + np.log(sigma2 / upper)
            else:
                sigma2 = upper + 1000.0
    else:
        sigma2 = np.maximum(lower, sigma2)
        sigma2 = np.where(sigma2 > upper, 
                        np.where(np.isinf(sigma2), upper + 1000.0, upper + np.log(sigma2 / upper)), 
                        sigma2)
    return sigma2

@jit(nopython=True, cache=True)
def compute_variance(params: np.ndarray, resids: np.ndarray, backcast: np.float64, var_bounds: np.ndarray) -> np.ndarray:
    if params.shape[-1] > 3:
        gjrgarch = True
    else:
        gjrgarch = False
    if params.ndim > 1:
        omega = params[:, 0]
        alpha = params[:, 1]
        if gjrgarch:
            gamma = params[:, 2]
            beta = params[:, 3]
        else:
            gamma = np.zeros(omega.shape)
            beta = params[:, 2]
        sigma2 = np.zeros((omega.shape[0], resids.shape[0]))
        sigma2[:, 0] = omega + (alpha + gamma/2 + beta) * backcast
        resids_squared = resids ** 2
        for t in range(1, resids.shape[0]):
            sigma2[:, t] = omega + alpha * resids_squared[t - 1] + beta * sigma2[:, t - 1]
            sigma2[:, t] += gamma * resids_squared[t - 1] if resids[t - 1] < 0 else np.zeros_like(omega, dtype=np.float64)
            sigma2[:, t] = bounds_check(sigma2[:, t], var_bounds[t])
    else:
        if gjrgarch:
            omega, alpha, gamma, beta = params
        else:
            omega, alpha, beta = params
            gamma = 0
        sigma2 = np.zeros(resids.shape)
        sigma2[0] = omega + (alpha + gamma/2 + beta) * backcast
        for t in range(1, resids.shape[0]):
            sigma2[t] = omega + alpha * (resids[t - 1] ** 2) + beta * sigma2[t - 1]
            sigma2[t] += gamma * resids[t - 1] ** 2 if resids[t - 1] < 0 else 0
            sigma2[t] = bounds_check(sigma2[t], var_bounds[t])
    return sigma2


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
