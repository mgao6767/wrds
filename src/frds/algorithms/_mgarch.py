import warnings
from typing import List, Tuple, Union
import itertools
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from frds.algorithms import GARCHModel, GJRGARCHModel

USE_CPP_EXTENSION = True

try:
    import frds.algorithms.utils.utils_ext as ext
except ImportError:
    USE_CPP_EXTENSION = False


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
        """Estimates the Multivariate GARCH(1,1)-CCC parameters via MLE

        Returns:
            Parameters: :class:`frds.algorithms.GARCHModel_CCC.Parameters`
        """
        if self.estimation_success:
            return self.parameters
        m1, m2 = self.model1, self.model2
        m1.fit()
        m2.fit()
        z1 = m1.resids / np.sqrt(m1.sigma2)
        z2 = m2.resids / np.sqrt(m2.sigma2)
        rho = np.corrcoef(z1, z2)[1, 0]
        m1_params = list(asdict(m1.parameters).values())[:-1]
        m2_params = list(asdict(m2.parameters).values())[:-1]
        starting_vals = [*m1_params, *m2_params, rho]

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


class GARCHModel_DCC(GARCHModel_CCC):
    """:doc:`/algorithms/garch-dcc` model with the following specification:

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
        a: float = np.nan
        b: float = np.nan
        loglikelihood: float = np.nan

    def __init__(
        self,
        returns1: Union[np.ndarray, GARCHModel],
        returns2: Union[np.ndarray, GARCHModel],
    ) -> None:
        """__init__

        Args:
            returns1 (Union[np.ndarray, GARCHModel]): ``(T,)`` array of ``T`` returns of first asset. Can also be an :class:`frds.algorithms.GARCHModel`.
            returns2 (Union[np.ndarray, GARCHModel]): ``(T,)`` array of ``T`` returns of second asset. Can also be an :class:`frds.algorithms.GARCHModel`.

        .. note::

            If ``returns`` is an array, it is best to be percentage returns for optimization.

            Estimated :class:`frds.algorithms.GARCHModel` can be used to save computation time.
        """
        if isinstance(returns1, np.ndarray):
            self.returns1 = np.asarray(returns1, dtype=np.float64)
            self.model1 = GARCHModel(self.returns1)
        if isinstance(returns2, np.ndarray):
            self.returns2 = np.asarray(returns2, dtype=np.float64)
            self.model2 = GARCHModel(self.returns2)
        if isinstance(returns1, GARCHModel):
            self.model1 = returns1
            self.returns1 = self.model1.returns
        if isinstance(returns2, GARCHModel):
            self.model2 = returns2
            self.returns2 = self.model2.returns
        self.estimation_success = False
        self.parameters = type(self).Parameters()

    def fit(self) -> Parameters:
        """Estimates the Multivariate GARCH(1,1)-DCC parameters via twp-step QML

        Returns:
            Parameters: :class:`frds.algorithms.GARCHModel_DCC.Parameters`
        """
        if self.estimation_success:
            return self.parameters
        m1, m2 = self.model1, self.model2
        m1.fit()
        m2.fit()
        a, b = self.starting_values()
        m1_params = list(asdict(m1.parameters).values())[:-1]
        m2_params = list(asdict(m2.parameters).values())[:-1]
        self.parameters = type(self).Parameters(*m1_params, *m2_params)
        starting_vals = [a, b]

        bounds = [
            # DCC parameters
            (0.0, 1.0),  # Bounds for a
            (0.0, 1.0),  # Bounds for b
        ]

        def a_plus_b_smaller_than_one(params: List[float]):
            a, b = params
            return 1.0 - (a + b)

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
                args=(),
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": a_plus_b_smaller_than_one},
                ],
            )
            if opt.success:
                self.estimation_success = True
                a, b = list(opt.x)
                self.parameters.a = a
                self.parameters.b = b
                self.parameters.loglikelihood=-opt.fun
        return self.parameters

    def starting_values(self) -> Tuple[float, float]:
        """Use a grid search to find the starting values for a and b

        Returns:
            Tuple[float, float]: [a, b]
        """
        def hierarchical_grid_search(loglikelihood_model, initial_range, base_grid_size=100, depth=2):
            current_range = initial_range
            best_params = None

            for i in range(depth):
                # 创建当前深度的网格
                a_grid = np.linspace(current_range[0, 0], current_range[0, 1], base_grid_size)
                b_grid = np.linspace(current_range[1, 0], current_range[1, 1], base_grid_size)
                a_values, b_values = np.meshgrid(a_grid, b_grid)
                parameters = np.stack([a_values.ravel(), b_values.ravel()], axis=1)
                
                # 过滤无效组合
                valid_indices = np.logical_and(parameters[:, 0] + parameters[:, 1] < 1,  # a + b < 1
                                               parameters[:, 0] >= 0.001,  # a >= 0.001
                                               parameters[:, 1] >= 0.001)  # b >= 0.001
                valid_parameters = parameters[valid_indices]
                
                # 计算对数似然值
                ll_values = -loglikelihood_model(valid_parameters)
                max_ll_index = np.argmax(ll_values)
                best_params = valid_parameters[max_ll_index]

                # 更新搜索范围为最佳参数周围的较小区域
                search_width = (current_range[:, 1] - current_range[:, 0]) / base_grid_size
                current_range = np.array([
                    [best_params[0] - search_width[0]*2, best_params[0] + search_width[0]*2],
                    [best_params[1] - search_width[1]*2, best_params[1] + search_width[1]*2]
                ])
            return best_params

        # 初始化参数范围
        initial_range = np.array([[0.001, 0.5], [0.6, 1]])  # [a_range, b_range]
        base_grid_size = 10  # 初始网格大小
        depth = 1  # 搜索深度，即细化的次数

        # 调用函数
        best_parameters = hierarchical_grid_search(self.loglikelihood_model, initial_range, base_grid_size, depth)
        initial_values = best_parameters
        return initial_values

    def loglikelihood_model(self, params: np.ndarray) -> float:
        """Calculates the negative log-likelihood based on the current ``params``.

        Args:
            params (np.ndarray): [a, b]

        Returns:
            float: negative log-likelihood
        """
        # fmt: off
        if len(params.shape)>1:
            a = params[:, 0]  # 第一列
            b = params[:, 1]  # 第二列
        else:
            a = params[0]
            b = params[1]
        resids1 = self.model1.resids
        resids2 = self.model2.resids
        sigma2_1 = self.model1.sigma2
        sigma2_2 = self.model2.sigma2
        # z1 and z2 are standardized residuals
        z1 = resids1 / np.sqrt(sigma2_1)
        z2 = resids2 / np.sqrt(sigma2_2)

        # The loglikelihood of the variance component (Step 1)
        l1 = self.model1.parameters.loglikelihood + self.model2.parameters.loglikelihood

        # The loglikelihood of the correlation component (Step 2)
        self.model1.fit()  # in case it was not estimated, no performance loss
        self.model2.fit()  # in case it was not estimated
        resids1 = self.model1.resids
        resids2 = self.model2.resids
        sigma2_1 = self.model1.sigma2
        sigma2_2 = self.model2.sigma2
        rho = conditional_correlations(resids1, resids2, sigma2_1, sigma2_2, a, b)
        # TODO: rare case rho is out of bounds
        rho = np.clip(rho, -0.99, 0.99)

        log_likelihood_terms = -0.5 * (
            - (z1**2 + z2**2)
            + np.log(1 - rho ** 2)
            + (z1 ** 2  + z2 ** 2  - 2 * rho * z1 * z2) / (1 - rho ** 2)
        )
        if len(params.shape)>1:
            l2 = np.sum(log_likelihood_terms, axis=1)
        else:
            l2 = np.sum(log_likelihood_terms)

        negative_loglikelihood = - (l1 + l2)
        return negative_loglikelihood

    def conditional_correlations(self, a: float, b: float) -> np.ndarray:
        """Computes the conditional correlations based on given a and b.
        Other parameters are

        Args:
            a (float): DCC parameter
            b (float): DCC parameter

        Returns:
            np.ndarray: array of conditional correlations
        """
        self.model1.fit()  # in case it was not estimated, no performance loss
        self.model2.fit()  # in case it was not estimated

        if USE_CPP_EXTENSION:
            return ext.dcc_conditional_correlation(
                a,
                b,
                self.model1.resids,
                self.model2.resids,
                self.model1.sigma2,
                self.model2.sigma2,
            )

        resids1 = self.model1.resids
        resids2 = self.model2.resids
        sigma2_1 = self.model1.sigma2
        sigma2_2 = self.model2.sigma2

        # z1 and z2 are standardized residuals
        z1 = resids1 / np.sqrt(sigma2_1)
        z2 = resids2 / np.sqrt(sigma2_2)
        Q_bar = np.corrcoef(z1, z2)
        q_11_bar, q_12_bar, q_22_bar = Q_bar[0, 0], Q_bar[0, 1], Q_bar[1, 1]
        T = len(z1)
        q11 = np.empty_like(z1)
        q12 = np.empty_like(z1)
        q22 = np.empty_like(z1)
        rho = np.zeros_like(z1)
        q11[0] = q_11_bar
        q22[0] = q_22_bar
        q12[0] = q_12_bar
        rho[0] = q12[0] / np.sqrt(q11[0] * q22[0])

        for t in range(1, T):
            q11[t] = (1 - a - b) * q_11_bar + a * z1[t - 1] ** 2 + b * q11[t - 1]
            q22[t] = (1 - a - b) * q_22_bar + a * z2[t - 1] ** 2 + b * q22[t - 1]
            q12[t] = (1 - a - b) * q_12_bar + a * z1[t - 1] * z2[t - 1] + b * q12[t - 1]
            rho[t] = q12[t] / np.sqrt(q11[t] * q22[t])

        return rho


class GJRGARCHModel_DCC(GARCHModel_DCC):
    """:doc:`/algorithms/gjr-garch-dcc` model with the following specification:

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
        gamma1: float = np.nan
        beta1: float = np.nan
        mu2: float = np.nan
        omega2: float = np.nan
        alpha2: float = np.nan
        gamma2: float = np.nan
        beta2: float = np.nan
        a: float = np.nan
        b: float = np.nan
        loglikelihood: float = np.nan

    def __init__(
        self,
        returns1: Union[np.ndarray, GJRGARCHModel],
        returns2: Union[np.ndarray, GJRGARCHModel],
    ) -> None:
        """__init__

        Args:
            returns1 (Union[np.ndarray, GJRGARCHModel]): ``(T,)`` array of ``T`` returns of first asset. Can also be an :class:`frds.algorithms.GJRGARCHModel`.
            returns2 (Union[np.ndarray, GJRGARCHModel]): ``(T,)`` array of ``T`` returns of second asset. Can also be an :class:`frds.algorithms.GJRGARCHModel`.

        .. note::

            If ``returns`` is an array, it is best to be percentage returns for optimization.

            Estimated :class:`frds.algorithms.GJRGARCHModel` can be used to save computation time.
        """
        if isinstance(returns1, np.ndarray):
            self.returns1 = np.asarray(returns1, dtype=np.float64)
            self.model1 = GJRGARCHModel(self.returns1)
        if isinstance(returns2, np.ndarray):
            self.returns2 = np.asarray(returns2, dtype=np.float64)
            self.model2 = GJRGARCHModel(self.returns2)
        if isinstance(returns1, GJRGARCHModel):
            self.model1 = returns1
            self.returns1 = self.model1.returns
        if isinstance(returns2, GJRGARCHModel):
            self.model2 = returns2
            self.returns2 = self.model2.returns
        self.estimation_success = False
        self.parameters = type(self).Parameters()

    def fit(self) -> Parameters:
        """Estimates the Multivariate GJR-GARCH(1,1)-DCC parameters via twp-step QML

        Returns:
            Parameters: :class:`frds.algorithms.GJRGARCHModel_DCC.Parameters`
        """
        return super().fit()

from numba import jit
@jit(nopython=True, cache=True)
def conditional_correlations(resids1, resids2, sigma2_1, sigma2_2, a, b) -> np.ndarray:
    # z1 and z2 are standardized residuals
    z1 = resids1 / np.sqrt(sigma2_1)
    z2 = resids2 / np.sqrt(sigma2_2)
    Q_bar = np.corrcoef(z1, z2)
    q_11_bar, q_12_bar, q_22_bar = Q_bar[0, 0], Q_bar[0, 1], Q_bar[1, 1]
    T = z1.shape[0]
    z1_squared = np.square(z1)
    z2_squared = np.square(z2)
    z1_z2 = np.multiply(z1, z2)
    coeff = 1 - a - b

    if isinstance(a, float):
        q11 = np.empty_like(z1)
        q12 = np.empty_like(z1)
        q22 = np.empty_like(z1)
        rho = np.zeros_like(z1)
        q11[0] = q_11_bar
        q22[0] = q_22_bar
        q12[0] = q_12_bar
        rho[0] = q12[0] / np.sqrt(q11[0] * q22[0])

        for t in range(1, T):
            q11[t] = coeff * q_11_bar + a * z1_squared[t - 1] + b * q11[t - 1]
            q22[t] = coeff * q_22_bar + a * z2_squared[t - 1] + b * q22[t - 1]
            q12[t] = coeff * q_12_bar + a * z1_z2[t - 1] + b * q12[t - 1]
        rho[1:] = q12[1:] / np.sqrt(q11[1:] * q22[1:])

    else:
        q11 = np.empty((a.shape[0], z1.shape[0]))
        q12 = np.empty((a.shape[0], z1.shape[0]))
        q22 = np.empty((a.shape[0], z1.shape[0]))
        rho = np.zeros((a.shape[0], z1.shape[0]))
        q11[:, 0] = q_11_bar
        q22[:, 0] = q_22_bar
        q12[:, 0] = q_12_bar
        rho[:, 0] = q12[:, 0] / np.sqrt(q11[:, 0] * q22[:, 0])

        for t in range(1, T):
            q11[:, t] = coeff * q_11_bar + a * z1_squared[t - 1] + b * q11[:, t - 1]
            q22[:, t] = coeff * q_22_bar + a * z2_squared[t - 1] + b * q22[:, t - 1]
            q12[:, t] = coeff * q_12_bar + a * z1_z2[t - 1] + b * q12[:, t - 1]
        rho[:, 1:] = q12[:, 1:] / np.sqrt(q11[:, 1:] * q22[:, 1:])

    return rho


if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint

    # df = pd.read_stata(
    #     "https://www.stata-press.com/data/r18/stocks.dta", convert_dates=["date"]
    # )
    df = pd.read_stata("~/Downloads/stocks.dta", convert_dates=["date"])

    df.set_index("date", inplace=True)
    # Scale returns to percentage returns for better optimization results
    toyota = df["toyota"].to_numpy() * 100
    nissan = df["nissan"].to_numpy() * 100
    honda = df["honda"].to_numpy() * 100

    model = GARCHModel_CCC(toyota, nissan)
    res = model.fit()
    pprint(res)

    dcc = GARCHModel_DCC(toyota, nissan)
    res = dcc.fit()
    pprint(res)

    toyota_garch = GARCHModel(toyota)
    nissan_garch = GARCHModel(nissan)

    dcc = GARCHModel_DCC(toyota_garch, nissan)
    res = dcc.fit()
    pprint(res)

    dcc = GARCHModel_DCC(toyota_garch, nissan_garch)
    res = dcc.fit()
    pprint(res)

    dcc = GJRGARCHModel_DCC(GJRGARCHModel(toyota), GJRGARCHModel(nissan))
    res = dcc.fit()
    pprint(res)
