from typing import Tuple
import numpy as np
from frds.algorithms import GJRGARCHModel, GJRGARCHModel_DCC
from frds.algorithms._mgarch import conditional_correlations

USE_CPP_EXTENSION = True
try:
    from frds.measures import measures_ext as ext
except ImportError:
    USE_CPP_EXTENSION = False


class LongRunMarginalExpectedShortfall:
    """:doc:`/measures/long_run_mes`"""

    def __init__(self, firm_returns: np.ndarray, market_returns: np.ndarray) -> None:
        """__init__

        Args:
            firm_returns (np.ndarray): ``(n_days,)`` array of the firm raw returns.
            market_returns (np.ndarray): ``(n_days,)`` array of the market raw returns.

        .. note::

           Raw returns should be used! They are automatically converted to log returns.
           Do NOT use percentage returns.
        """
        # Convert raw returns to log returns
        self.firm_returns = np.log(1 + np.asarray(firm_returns, dtype=np.float64))
        self.market_returns = np.log(1 + np.asarray(market_returns, dtype=np.float64))
        assert self.firm_returns.shape == self.market_returns.shape
        # Scale to percentage (log) returns
        # This is for better (GJR)GARCH estimation
        self.firm_returns *= 100
        self.market_returns *= 100
        self.firm_model = GJRGARCHModel(self.firm_returns)
        self.market_model = GJRGARCHModel(self.market_returns)
        self.dcc_model = GJRGARCHModel_DCC(self.firm_model, self.market_model)

    def estimate(self, h=22, S=10_000, C=-0.1, random_seed=42) -> float:
        """h-step-ahead LRMES forecasts conditional on a systemic event of market decline C

        Args:
            h (int, optional): h-period-ahead prediction horizon. Defaults to 22.
            S (int, optional): sample size used in simulation to generate LRMES forecasts. Defaults to 10000.
            C (float, optional): market decline used to define systemic event. Defaults to -0.1, i.e. -10%.
            random_seed (int, optional): random seed. Defaults to 42.

        Returns:
            float: the firm's LRMES forecast
        """
        rng = np.random.default_rng(random_seed)

        # Estimate the (GJR)GARCH-DCC model
        self.dcc_model.fit()
        if not self.dcc_model.estimation_success:
            raise RuntimeError("(GJR)GARCH-DCC model estimation failed!")
        # Construct GJR-GARCH-DCC standardized innovations for the sample
        # See the step 1 of Computing LRMES section
        firm_variances = self.firm_model.sigma2
        firm_resids = self.firm_model.resids
        market_variances = self.market_model.sigma2
        market_resids = self.market_model.resids
        # Conditional correlations
        a, b = self.dcc_model.parameters.a, self.dcc_model.parameters.b
        self.dcc_model.model1.fit()  # in case it was not estimated, no performance loss
        self.dcc_model.model2.fit()  # in case it was not estimated
        resids1 = self.dcc_model.model1.resids
        resids2 = self.dcc_model.model2.resids
        sigma2_1 = self.dcc_model.model1.sigma2
        sigma2_2 = self.dcc_model.model2.sigma2
        rho = conditional_correlations(resids1, resids2, sigma2_1, sigma2_2, a, b)
        # Standarized residuals
        z_m = market_resids / np.sqrt(market_variances)
        z_i = firm_resids / np.sqrt(firm_variances)
        # Firm shock orthogonal to market
        xi_i = (z_i - rho * z_m) / np.sqrt(1 - rho**2)
        sample = np.array([xi_i, z_m])

        # Sample with replacement S*h innovations
        # sample = sample.T[rng.choice(sample.shape[1], (S, h), replace=True)]

        idx = rng.choice(len(xi_i), size=(S, h), replace=True)
        # 使用高级索引和广播来生成 sample 数组
        sample = np.zeros((S, h, 2))
        sample[:, :, 0] = xi_i[idx]
        sample[:, :, 1] = z_m[idx]

        assert sample.shape == (S, h, 2)

        Q_bar = np.corrcoef(z_i, z_m)

        systemic_event_firm_return = self.simulation(
            innovation=sample,
            C=C,
            firm_var=self.firm_model.sigma2[-1],
            mkt_var=self.market_model.sigma2[-1],
            firm_resid=self.firm_model.resids[-1],
            mkt_resid=self.market_model.resids[-1],
            a=self.dcc_model.parameters.a,
            b=self.dcc_model.parameters.b,
            rho=rho[-1],
            Q_bar=Q_bar,
        )
        systemic_event_firm_return = systemic_event_firm_return[systemic_event_firm_return != False]

        # 计算这些值的均值
        return -np.mean(systemic_event_firm_return)

    def simulation(
        self,
        innovation: np.ndarray,
        C: float,
        firm_var: float,
        mkt_var: float,
        firm_resid: float,
        mkt_resid: float,
        a: float,
        b: float,
        rho: float,
        Q_bar: np.ndarray,
    ) -> Tuple[float, bool]:
        """A simulation to compute the firm's return given the parameters.
        This method should be used internally.

        Args:
            innovation (np.ndarray): ``(h,2)`` array of market and firm innovations
            C (float): market decline used to define systemic event. Defaults to -0.1, i.e. -10%.
            firm_var (float): the firm conditional variance at time :math:`T`, used as starting value in forecast
            mkt_var (float): the market conditional variance at time :math:`T`, used as starting value in forecast
            firm_resid (float): the firm residual at time :math:`T`, used as starting value in forecast
            mkt_resid (float): the market residual at time :math:`T`, used as starting value in forecast
            a (float): DCC parameter
            b (float): DCC parameter
            rho (float): the last conditional correlation at time :math:`T`, used as starting value in forecast
            Q_bar (np.ndarray): ``(2,2)`` array of sample correlation matrix of standarized residuals

        Returns:
            Tuple[float, bool]: tuple of the firm return and whether a systemic event occurs
        """
        q_i_bar = Q_bar[0, 0]
        q_m_bar = Q_bar[1, 1]
        q_im_bar = Q_bar[1, 0]
        S, h, d = innovation.shape
        q_i = np.full(S, 1)
        q_m = np.full(S, 1)
        q_im = np.full(S, rho)

        pi = self.firm_model.parameters
        mu_i = pi.mu

        # omega_i, alpha_i, gamma_i, beta_i = pi.omega, pi.alpha, pi.gamma, pi.beta
        omega_i = np.full(S, pi.omega)
        alpha_i = np.full(S, pi.alpha)
        gamma_i = np.full(S, pi.gamma)
        beta_i = np.full(S, pi.beta)

        pm = self.market_model.parameters
        mu_m = pm.mu
        # omega_m, alpha_m, gamma_m, beta_m = pm.omega, pm.alpha, pm.gamma, pm.beta
        omega_m = np.full(S, pm.omega)
        alpha_m = np.full(S, pm.alpha)
        gamma_m = np.full(S, pm.gamma)
        beta_m = np.full(S, pm.beta)

        firm_return = np.empty((S, h))
        mkt_return = np.empty((S, h))

        # lagged residuals
        # resid_i_tm1 = firm_resid
        # resid_m_tm1 = mkt_resid
        resid_i_tm1 = np.full(S, firm_resid)
        resid_m_tm1 = np.full(S, mkt_resid)
        # lagged conditonal variance
        # firm_var_tm1 = firm_var
        # mkt_var_tm1 = mkt_var
        firm_var_tm1 = np.full(S, firm_var)
        mkt_var_tm1 = np.full(S, mkt_var)
        # lagged standarized residuals
        # firm_innov_tm1 = resid_i_tm1 / np.sqrt(firm_var_tm1)
        # mkt_innov_tm1 = resid_m_tm1 / np.sqrt(mkt_var_tm1)
        firm_innov_tm1 = np.full(S, firm_resid / np.sqrt(firm_var))
        mkt_innov_tm1 = np.full(S, mkt_resid / np.sqrt(mkt_var))
        coff = (1 - a - b)

        for h in range(innovation.shape[-2]):
            # fmt: off
            # Each iteration is a one-step-ahead forecast
            # conditional variance this time
            firm_var_t = omega_i + alpha_i * (resid_i_tm1**2) + beta_i * firm_var_tm1
            # firm_var_t += gamma_i * (resid_i_tm1**2) if resid_i_tm1 < 0 else 0
            firm_var_t += np.where(resid_i_tm1 < 0, gamma_i * (resid_i_tm1**2), 0)

            mkt_var_t = omega_m + alpha_m * (resid_m_tm1**2) + beta_m * mkt_var_tm1
            # mkt_var_t += gamma_m * (resid_m_tm1**2) if resid_m_tm1 < 0 else 0
            mkt_var_t += np.where(resid_m_tm1 < 0, gamma_m * (resid_m_tm1**2), 0)

            # conditional correlation this time
            q_i = coff * q_i_bar + a * firm_innov_tm1**2 + b * q_i
            q_m = coff * q_m_bar + a * mkt_innov_tm1**2 + b * q_m
            q_im = coff * q_im_bar + a * firm_innov_tm1*mkt_innov_tm1 + b * q_im
            rho_h = q_im / np.sqrt(q_i * q_m)

            # innovations this time
            firm_innov = innovation[:, h, 0]
            mkt_innov = innovation[:, h, 1]

            # market excess return
            # or, residual this time, conditional volatility * innovation (z_m)
            epsilon_m = np.sqrt(mkt_var_t) * mkt_innov
            # Beta of the firm = cov/var(mkt)
            # Beta = rho_h * np.sqrt(firm_var_t) / np.sqrt(mkt_var_t)
            # epsilon_i = Beta * epsilon_m

            epsilon_i = np.sqrt(firm_var_t) * (
                rho_h * mkt_innov + np.sqrt(1 - rho_h**2) * firm_innov
            )

            mkt_return[:, h] = mu_m + epsilon_m
            firm_return[:, h] = mu_i + epsilon_i

            firm_var_tm1 = firm_var_t
            mkt_var_tm1 = mkt_var_t
            resid_i_tm1 = epsilon_i
            resid_m_tm1 = epsilon_m
            firm_innov_tm1 = resid_i_tm1 / np.sqrt(firm_var_tm1)
            mkt_innov_tm1 = mkt_innov

        # Convert back to original scale
        mkt_return /= 100
        firm_return /= 100

        # systemic event if over the prediction horizon,
        # the market falls by more than C
        systemic_event = np.exp(np.sum(mkt_return, axis=1)) - 1 < C
        # no need to simulate firm returns if there is no systemic event
        return np.where(systemic_event, np.exp(np.sum(firm_return, axis=1)) - 1, False)
