##############################################
 Long-Run Marginal Expected Shortfall (LRMES)
##############################################

.. tip:: 

   Check `Examples`_ section for code guide and comparison to NYU's 
   `V-Lab <https://vlab.stern.nyu.edu/srisk/RISK.USFIN-MR.MESSIM>`_, 

**************
 Introduction
**************

Long-run Marginal Exepected Shortfall (LRMES) measures the expected firm
return conditional on a systemic event, which is a market decline below
a threshold :math:`C` over a time horizon :math:`h`.

If the multiperiod arithmetic market return between :math:`t+1` to
:math:`t+h` is :math:`R_{mt+1:t+h}`, then the systemic event is
:math:`\{R_{mt+1:t+h}<C\}`.

:math:`LRMES_{it}` for firm :math:`i` at time :math:`t` is then defined
as

.. math::

   LRMES_{it} = -E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]

`Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_ use a
:doc:`/algorithms/gjr-garch-dcc` model to construct the LRMES
predictions.

.. tip::

   If you're new to econommetrics, I suggest the following sequence of
   readings.

   Univariate GARCH models:

   #. :doc:`/algorithms/garch`, Generalized Autoregressive Conditional
      Heteroskedasticity modelling.
   #. :doc:`/algorithms/gjr-garch` extends GARCH to take into account
      asymmetric effect of negative news.

   Multivariate GARCH models:

   #. :doc:`/algorithms/garch-ccc` captures both conditional variances
      and conditioal covariances, assuming constant correlation.
   #. :doc:`/algorithms/garch-dcc` further allows for time-varying
      correlations.

   Finally,

   #. :doc:`/algorithms/gjr-garch-dcc` combines the above.

*************************
 Bivariate GJR-GARCH-DCC
*************************

Here, I dicuss the bivariate case in the context of computing
:doc:`/measures/srisk`, where :math:`i` index the firm and :math:`m`
refers to the market.

Return series
=============

Let firm and market log returns be :math:`r_{it} = \log(1 + R_{it})` and
:math:`r_{mt} = \log(1 + R_{mt})`. Conditional on the information set
:math:`\mathcal{F}_{t-1}` available at time :math:`t-1`, the return pair
has an (unspecified) distribution :math:`\mathcal{D}` with zero mean and
time-varying covariance,

.. math::
   :label: eq:garch_dcc

    \begin{bmatrix}r_{it} \\ r_{mt}\end{bmatrix} | \mathcal F_{t-1} \sim \mathcal D\left(\mathbf 0, \begin{bmatrix}\sigma_{it}^2 & \rho_{it}\sigma_{it}\sigma_{mt} \\\\ \rho_{it}\sigma_{it}\sigma_{mt} & \sigma_{mt}^2 \end{bmatrix}\right)

Alternatively, I present below the constant mean form of return series,

.. math::
   :label: return_eqn_1

   r_{it} = \mu_i + \epsilon_{it}

.. math::
   :label: return_eqn_2

   r_{mt} = \mu_m + \epsilon_{mt}

where,

-  :math:`\mu=0` assumes zero-mean. One can also assume a constant
   mean or even other structure.

-  :math:`\epsilon_{it} = \sigma_{it} z_{it}`, :math:`\epsilon_{mt} =
   \sigma_{mt} z_{mt}`, where :math:`\sigma_{it}` is the conditional
   volatility at time :math:`t`.

-  :math:`z_{it}, z_{mt}` are standarized residuals that follow an unknown bivariate distribution with
   zero mean and maybe some covariance structure. However, generally we can assume :math:`z_{it}` and :math:`z_{mt}` are i.i.d. 
   standard normal

.. important::

   In a DCC-GARCH model, assuming that the shock innovation :math:`z` is
   standard normally distributed only implies that the conditional
   innovations are normal given the past information. It does **not**
   imply that the returns themselves are normally distributed.

   The returns :math:`r` in a GARCH framework are often modeled as:

   .. math::

      r_t = \mu_t + \sigma_t \cdot z_t

   where :math:`\mu_t` is the conditional mean, :math:`\sigma_t` is the
   conditional volatility, and :math:`z_t` is the innovation term.

   -  :math:`\mu_t` could be modeled in many ways, perhaps as a function
      of past returns or other variables, and this will affect the
      distribution of :math:`r_t`.

   -  :math:`\sigma_t` is also a function of past volatilities and
      innovations, and hence varies over time, even if :math:`z_t` is
      standard normal.

   So, while :math:`z_t` might be standard normal, :math:`\sigma_t \cdot
   z_t` is normal but not standard normal, and :math:`\mu_t + \sigma_t
   \cdot z_t` may not be normal at all depending on the form of
   :math:`\mu_t`. Therefore, the assumption about :math:`z` being
   standard normal is about the standardized returns or innovations, not
   the raw returns :math:`r`. **This is the reason why the return pair
   has an (unspecified) distribution** :math:`\mathcal{D}`.

Conditional variance
====================

Specifically, the GJR-GARCH models the conditional variances as:

.. math::
   :label: eq:gjr_garch

   \begin{align*}
   \sigma_{it}^2 &= \omega_{i} + \alpha_{i} \epsilon^2_{it-1} + \gamma_{i} \epsilon^2_{it-1} I^-_{it-1} + \beta_{i} \sigma^2_{it-1}, \\\\
   \sigma_{mt}^2 &= \omega_{m} + \alpha_{m} \epsilon^2_{mt-1} + \gamma_{m} \epsilon^2_{mt-1} I^-_{mt-1} + \beta_{m} \sigma^2_{mt-1}
   \end{align*}

where :math:`I^-_{it} = 1` if :math:`r_{it} < 0` and :math:`I^-_{mt} =
1` if :math:`r_{mt} < 0`.

.. note::

   `Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_ use a 
   zero-mean assumption, so that :math:`r` is used in the above equation 
   instead of the residual :math:`\epsilon`. They use :math:`\epsilon` to denote
   the standardized residual, which in this note is :math:`z` to be consistent
   with other notes of ``frds``.

Dynamic correlation
===================

With these individual variances, the **Dynamic Conditional Correlation
(DCC)** models the covariance matrix :math:`\mathbf{H}_t` as:

.. math::
   :label: conditional_cov_matrix_modified

       \mathbf{H}_t = \begin{pmatrix}
       \sigma^2_{it} & \rho_{it}\sigma_{it} \sigma_{mt} \\\\
       \rho_{it}\sigma_{it} \sigma_{mt} & \sigma^2_{mt}
       \end{pmatrix} = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t

where :math:`\mathbf{D}_t=\begin{bmatrix}\sigma_{it} & 0 \\\\ 0 &
\sigma_{mt}\end{bmatrix}`, and :math:`\mathbf{R_t}=\begin{bmatrix}1 &
\rho_t \\\\ \rho_t & 1\end{bmatrix}` is the correlation matrix of the
volatility-adjusted returns :math:`z = (r-\mu)/\sigma` (standardized
residuals from the return model).

A proxy process for R
=====================

As discussed in :doc:`/algorithms/gjr-garch-dcc`, Engle (2002) models
the time-varying :math:`\mathbf{R}_t` via a proxy process
:math:`\mathbf{Q}_t`.

.. math::
   :label: eq:dcc_corr

   \mathbf{R}_t = \text{Corr}\begin{pmatrix}
   z_{it} \\\\
   z_{mt}
   \end{pmatrix}
   = \begin{bmatrix}
   1 & \rho_{it} \\\\
   \rho_{it} & 1
   \end{bmatrix}
   = \text{diag}(\mathbf{Q}_{it})^{-1/2} \mathbf{Q}_{it} \text{diag}(\mathbf{Q}_{it})^{-1/2}

where :math:`\mathbf{Q}_{it}` is the so-called pseudo correlation
matrix.

.. math::
   :label: eq:q_matrix

   \mathbf{Q}_{it} = \begin{bmatrix}
   q_{it} & q_{imt} \\\\
   q_{imt} & q_{mt}
   \end{bmatrix}

The DCC then models the dynamics of :math:`\mathbf{Q}_{it}` as

.. math::
   :label: eq:dcc_dynamics

   \mathbf{Q}_{it} = (1-a-b) \bar{\mathbf{Q}}_i + a \mathbf{z}_{t-1} \mathbf{z}'_{t-1} + b \mathbf{Q}_{it-1}

where,

-  :math:`\bar{\mathbf{Q}}_i` is the unconditional correlation matrix of
   the firm and market adjusted returns (standardized residuals), and

-  :math:`\mathbf{z}_{t-1} = \begin{bmatrix}\epsilon_{it-1}/\sigma_{it}
   \\\\ \epsilon_{mt-1}/\sigma_{mt} \end{bmatrix} =
   \begin{bmatrix}(r_{it-1}-\mu_i)/\sigma_{it} \\\\
   (r_{mt-1}-\mu_m)/\sigma_{mt} \end{bmatrix}`

The Q process is updated as follows:

.. math::
   :label: q11

   q_{it} = (1 - a - b) \overline{q}_{i} + a z^2_{i,t-1} + b q_{i,t-1}

.. math::
   :label: q22

   q_{mt} = (1 - a - b) \overline{q}_{m} + a z^2_{m,t-1} + b q_{m,t-1}

.. math::
   :label: q12

   q_{imt} = (1 - a - b) \overline{q}_{im} + a z_{i,t-1} z_{m,t-1} + b q_{im,t-1}

The dynamic conditional correlation :math:`\rho_t` is given by:

.. math::
   :label: rho_t

   \rho_{it} = \frac{q_{imt}}{\sqrt{q_{it} q_{mt}}}

Estimating GJR-GARCH-DCC
========================

The above model is typically estimated by a two-step QML estimation
procedure. More extensive details on this modeling approach and
estimation are provided in `Engle (2009)
<http://www.jstor.org/stable/j.ctt7sb6w>`_.

.. note::

   Equation 4.33 in `Engle (2009)
   <http://www.jstor.org/stable/j.ctt7sb6w>`_ shows that the log
   likelihood can be additively divided into two parts, one concerns the
   variance and the other concerns correlation. Therefore, we can solve
   for the variance and correlation parameters in two separate steps,
   hence "two-step" QML.

   See the loglikelihood function section of
   :doc:`/algorithms/gjr-garch-dcc` for more.

Specifically, we first estimate a GJR-GARCH(1,1) for each firm (and
market)'s log return series to obtain the conditional volatilities
:math:`\sigma` and hence :math:`z = (r-\mu)/\sigma`. In the second step,
we use the estimated coefficients to estimate the DCC model for each
pair of firm returns and market returns.

This is done via :class:`frds.algorithms.GJRGARCHModel_DCC`.

*****************
 Computing LRMES
*****************

LRMES is computed via a simulation approach. Appendix A. Simulation
Algorithm for LRMES in `Engle (2009)
<http://www.jstor.org/stable/j.ctt7sb6w>`_ describes the exact steps to
construct LRMES forecasts.

.. tip::

   The general idea is to simulate market returns and use the estimated
   GJR-GARCH-DCC model to derive the corresponding firm returns. We then
   use the distribution of returns to estimate the firm's LRMES.

**Step 1**. Construct GJR-GARCH-DCC standardized innovations for the
training sample :math:`t=1,...,T`, where :math:`\xi_{it}` is
standardized, linearly orthogonal shocks of the firm to the market on
day :math:`t`,

.. math::
   :label: eq:step1

   \begin{align}
    z_{mt} &= \frac{r_{mt}-\mu_m}{\sigma_{mt}} \\\\
    \xi_{it} &= \left(\frac{r_{it}-\mu_{i}}{\sigma_{it}} - \rho_{it} z_{mt}\right) / \sqrt{1-\rho^2_{it}}
   \end{align}

.. note::

   Given the conditional correlation :math:`\rho` between the
   standardized residuals :math:`z_i` and :math:`z_m` from a bivariate
   (GJR)GARCH model under Dynamic Conditional Correlation (DCC), we can
   compute :math:`z_i` given :math:`z_m` using the following formula:

   .. math::

      z_i = \rho \times z_m + \sqrt{1 - \rho^2} \times \xi

   Here, :math:`\xi` is a standard normal random variable that is
   independent of :math:`z_m`, 

   .. math::

      \xi = (z_i - \rho z_m) / \sqrt{1 - \rho^2}

**Step 2**. Sample with replacement :math:`S\times h` pairs of
:math:`[\xi_{it}, z_{mt}]'`, which are used as the simulated
innovations from time :math:`T+1` to :math:`T+h`. Notice that in the
algorithm, the innovations are simulated by resampling the standardized
residuals of the GARCH-DCC rather than relying on parametric
assumptions.

.. note::
   
   Basically, we need residuals :math:`\epsilon_{mT+t}=\sigma_{mT+t}z_{mT+t}` 
   and :math:`\epsilon_{iT+t}=\sigma_{iT+t}z_{iT+t}` for :math:`t=1,...,h`.

   Requirements are
   
   #. The standarized residuals :math:`z_{iT+t}` and :math:`z_{mT+t}` are i.i.d. normal.
   #. :math:`\epsilon_{mT+t}` and :math:`\epsilon_{iT+t}` have a conditional correlation :math:`\rho_t`.

   Given conditional variance, conditional correlation, :math:`z_{iT+t}` has to be

   .. math::

      z_{iT+t} = \rho_t \times z_{mT+t} + \sqrt{1 - \rho_t^2} \times \xi_{iT+t}

   We use the sampled (bootstrapped) :math:`h` pairs of :math:`[\xi_{it}, z_{mt}]'` 
   as :math:`[\xi_{iT+t}, z_{mT+t}]'` for :math:`t=1,...,h`.

**Step 3**. Use the pseudo sample of innovations as inputs of the DCC
and (GJR)GARCH filters, respectively. Initial conditions are the last values
of the conditional correlation :math:`\rho_{iT}` and variances
:math:`\sigma^2_{iT}` and :math:`\sigma^2_{mT}`. This step delivers
:math:`S` pseudo samples of (GJR)GARCH-DCC logarithmic returns from period
:math:`T+1` to period :math:`T+h`, conditional on the realized process
up to time :math:`T`, that is

.. math::
   :label: eq:step3

   \begin{bmatrix}
      r^s_{iT+t} \\\\
      r^s_{mT+t}
   \end{bmatrix}_{t=1,...,h} | \mathcal{F}_{T}

Specifically, in a simulation :math:`s` with simulated innovations :math:`[\xi^s_{iT+t}, z^s_{mT+t}]'`,
for the 1-step-ahead prediction, compute :math:`\hat{\sigma}^2_{iT+1}` and :math:`\hat{\sigma}^2_{mT+1}`,

.. math:: 

   \begin{align}
   \hat{\sigma}^2_{iT+1} &= \omega_i + \left[\alpha_i+\gamma_i I(\epsilon_{iT}<0)\right] \epsilon_{iT}^2 + \beta_i {\sigma}^2_{iT} \\\\
   \hat{\sigma}^2_{mT+1} &= \omega_m + \left[\alpha_m+\gamma_m I(\epsilon_{mT}<0)\right] \epsilon_{mT}^2 + \beta_m {\sigma}^2_{mT} 
   \end{align}

where, :math:`\sigma^2_{iT}` and :math:`\sigma^2_{mT}` are the last conditional 
variances, :math:`\epsilon_{iT}` and :math:`\epsilon_{mT}` are the last residuals. All of these are known.

The updated :math:`Q_{T+1}` is given by 

.. math::

   \begin{align}
   \hat{q}_{iT+1} &= (1 - a - b) \overline{q}_{i} + a z^2_{iT} + b {q}_{iT} \\\\
   \hat{q}_{mT+1} &= (1 - a - b) \overline{q}_{m} + a z^2_{mT} + b {q}_{mT} \\\\
   \hat{q}_{imT+1} &= (1 - a - b) \overline{q}_{im} + a z_{iT} z_{mT} + b {q}_{imT}
   \end{align}

where :math:`z_{iT}` and :math:`z_{mT}` are the last standardized residuals. 
:math:`q_{iT}`, :math:`q_{mT}` and :math:`q_{imT}` are from the last :math:`Q_T`.
All of these are known.

The 1-step-ahead conditional correlation :math:`\hat{\rho}_{iT+1}` is given by:

.. math::

   \hat{\rho}_{iT+1} = \frac{\hat{q}_{imT+1}}{\sqrt{\hat{q}_{iT+1} \hat{q}_{mT+1}}}

This conditional correlation :math:`\hat{\rho}_{iT+1}` is then used to compute the 1-step-ahead returns 
given the 1-step-ahead forecast of conditional variances :math:`\hat{\sigma}^2_{iT+1}` and :math:`\hat{\sigma}^2_{mT+1}`,
and innovations :math:`[\xi^s_{iT+1}, z^s_{mT+1}]'`,

.. math::

   \begin{align}
   \hat{r}_{mT+1} &= \mu_m + \hat{\epsilon}_{mT+1} = \mu_m + \hat{\sigma}_{mT+1} z^s_{mT+1} \\\\
   \hat{r}_{iT+1} &= \mu_i + \hat{\epsilon}_{iT+1} = \mu_i + \hat{\sigma}_{iT+1} (\hat{\rho}_{iT+1} z^s_{mT+1} + \sqrt{1-\hat{\rho}^2_{iT+1}} \xi^s_{iT+1})
   \end{align}

Then, for :math:`h>1`, we use the :math:`h-1` forecasts as inputs.

.. math:: 

   \begin{align}
   \hat{\sigma}^2_{iT+h} &= \omega_i + \left[\alpha_i+\gamma_i I(\hat{\epsilon}_{iT+h-1}<0)\right] \hat{\epsilon}_{iT+h-1}^2 + \beta_i \hat{\sigma}^2_{iT+h-1} \\\\
   \hat{\sigma}^2_{mT+h} &= \omega_m + \left[\alpha_m+\gamma_m I(\hat{\epsilon}_{mT+h-1}<0)\right] \hat{\epsilon}_{mT+h-1}^2 + \beta_m \hat{\sigma}^2_{mT+h-1} 
   \end{align}

where,

.. math::

   \begin{align}
   \hat{\epsilon}_{mT+h-1} &= \hat{\sigma}_{mT+h-1} z^s_{mT+h-1} \\\\
   \hat{\epsilon}_{iT+h-1} &= \hat{\sigma}_{iT+h-1} \left[\hat{\rho}_{iT+h-1} z^s_{mT+h-1} + \sqrt{1-\hat{\rho}^2_{iT+h-1}} \xi^s_{iT+h-1}\right]
   \end{align}

Then, update DCC coefficients,

.. math::

   \begin{align}
   \hat{q}_{iT+h} &= (1 - a - b) \overline{q}_{i} + a {\left(\frac{\hat{\epsilon}_{iT+h-1}}{\hat{\sigma}_{iT+h-1}}\right)}^2 + b \hat{q}_{i,T+h-1} \\\\
   \hat{q}_{mT+h} &= (1 - a - b) \overline{q}_{m} + a {\left(\frac{\hat{\epsilon}_{mT+h-1}}{\hat{\sigma}_{mT+h-1}}\right)}^2 + b \hat{q}_{m,T+h-1} \\\\
   \hat{q}_{imT+h} &= (1 - a - b) \overline{q}_{im} + a \left(\frac{\hat{\epsilon}_{mT+h-1}}{\hat{\sigma}_{mT+h-1}}\frac{\hat{\epsilon}_{iT+h-1}}{\hat{\sigma}_{iT+h-1}}\right) + b \hat{q}_{im,T+h-1}
   \end{align}

The dynamic conditional correlation :math:`\hat{\rho}_{iT+h}` is given by:

.. math::

   \hat{\rho}_{iT+h} = \frac{\hat{q}_{imT+h}}{\sqrt{\hat{q}_{iT+h} \hat{q}_{mT+h}}}

This conditional correlation :math:`\hat{\rho}_{iT+h}` is then used to compute the h-step-ahead returns 
given the h-step-ahead forecast of conditional variances :math:`\hat{\sigma}^2_{iT+h}` and :math:`\hat{\sigma}^2_{mT+h}`,
and innovations :math:`[\xi^s_{iT+h}, z^s_{mT+h}]'`,

.. math::

   \begin{align}
   \hat{r}_{mT+h} &= \mu_m + \hat{\sigma}_{mT+h} z^s_{mT+h} \\\\
   \hat{r}_{iT+h} &= \mu_i + \hat{\sigma}_{iT+h} (\hat{\rho}_{iT+h} z^s_{mT+h} + \sqrt{1-\hat{\rho}^2_{iT+h}} \xi^s_{iT+h})
   \end{align}

So we have in this simulation
:math:`s` a set of market and firm (log) returns, :math:`r^s_{iT+t}` and :math:`r^s_{mT+t}`, :math:`t=1,\dots,h`.

**Step 4**. Construct the multiperiod arithmetic firm (market) return of
each pseudo sample,

.. math::
   :label: eq:returns_sim

   R^s_{iT+1:T+h} = \exp \left(\sum_{t=1}^{h} r^s_{iT+t} \right) -1

**Step 5**. Compute LRMES as the Monte Carlo average of the simulated
multiperiod arithmetic returns conditional on the systemic event,

.. math::
   :label: eq:lrmes_est

   LRMES_{iT} = - \frac{\sum_{s=1}^S R^s_{iT+1:T+h} I(R^s_{mT+1:T+h}<C)}{\sum_{s=1}^S I(R^s_{mT+1:T+h}<C)}

.. tip::
   This LRMES estimate can then be used to compute :doc:`/measures/srisk`.

************
 References
************

-  `Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_,
   *SRISK: A Conditional Capital Shortfall Measure of Systemic Risk*,
   Review of Financial Studies, 30 (1), 48–79.

-  `Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993)
   <https://doi.org/10.1111/j.1540-6261.1993.tb05128.x>`_, "On the
   Relation Between the Expected Value and the Volatility of the Nominal
   Excess Return on Stocks." *The Journal of Finance*, 48(5), 1779-1801.

-  `Engle, R. (2002) <https://www.jstor.org/stable/1392121>`_, "Dynamic
   Conditional Correlation: A Simple Class of Multivariate Generalized
   Autoregressive Conditional Heteroskedasticity Models." *Journal of
   Business & Economic Statistics*, 20(3), 339-350.

*****
 API
*****

.. autoclass:: frds.measures.LRMES

.. autoclass:: frds.measures.LongRunMarginalExpectedShortfall

**********
 Examples
**********

Download daily price data for GS and SP500 and compute returns.

>>> from frds.measures import LRMES
>>> import yfinance as yf
>>> import numpy as np
>>> data = yf.download(['GS', '^GSPC'], start='1999-01-01', end='2022-12-31')['Adj Close']
>>> data['GS'] = data['GS'].pct_change()
>>> data['^GSPC'] = data['^GSPC'].pct_change()
>>> data = data.dropna()

Compute LRMES, conditioal on a 40% market decline over the next six months.

>>> lrmes = LRMES(data['GS'], data['^GSPC'])
>>> lrmes.estimate(S=10000, h=22*6, C=-0.4)
0.3798383812883635

Additionally, below is an example output for estimating the daily LRMES for GS
using parallel computing.

.. image:: /images/GS_LRMES.png

.. note:: 
   
   These estimates are similar to NYU's 
   `V-Lab <https://vlab.stern.nyu.edu/srisk/RISK.USFIN-MR.MESSIM>`_, 
   which is more stable and on average higher. V-Lab's simulated LRMES is 
   typically above 40%.

   Possible reasons include differences in
   
   #. the training sample  
   #. the (GJR)GARCH-DCC model specification and estimation, where V-Lab uses 
      a zero-mean return model but I use a constant mean
   #. the number of simulations, sampling of past residuals, etc. 


This is computed using the following code.

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   from frds.measures import LRMES
   import numpy as np
   import pandas as pd
   import yfinance as yf
   import matplotlib.pyplot as plt


   def compute_lrmes_for_date(args):
       date, data = args
       if date.year < 2001:
           return None
       sub_data = data.loc[:date]
       lrmes = LRMES(sub_data["GS"], sub_data["^GSPC"]).estimate(h=22 * 6, C=-0.4)
       print((date, lrmes))
       return (date, lrmes)


   if __name__ == "__main__":
       # Download daily price data for GS and SP500
       data = yf.download(["GS", "^GSPC"], start="1994-01-01", end="2022-12-31")["Adj Close"]

       data["GS"] = data["GS"].pct_change()
       data["^GSPC"] = data["^GSPC"].pct_change()
       data = data.dropna()

       data["GS_Price"] = (1 + data["GS"]).cumprod() * 100
       data["^GSPC_Price"] = (1 + data["^GSPC"]).cumprod() * 100

       with ProcessPoolExecutor() as executor:
           lrmes_values = list(
               executor.map(
                   compute_lrmes_for_date,
                   [(d, data) for d in data.index.unique()],
               )
           )

       lrmes_values = [x for x in lrmes_values if x is not None]

       lrmes_df = pd.DataFrame(lrmes_values, columns=["Date", "LRMES"]).set_index("Date")

       data = pd.merge_asof(data, lrmes_df, left_index=True, right_index=True, direction="backward")

       model = LRMES(data["GS"], data["^GSPC"])
       model.dcc_model.fit()
       data["GS Volatility"] = np.sqrt(model.dcc_model.model1.sigma2)
       data["S&P500 Volatility"] = np.sqrt(model.dcc_model.model2.sigma2)

       fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

       # Plot data
       axs[0].plot(data["GS"], label="GS data", color="blue")
       axs[0].plot(data["^GSPC"], label="SP500 data", color="red")
       axs[0].set_title("Daily Returns")
       axs[0].set_ylabel("Returns")
       axs[0].legend()
       axs[0].grid(True)

       axs[1].plot(data["GS_Price"], label="GS Indexed Price", color="blue")
       axs[1].plot(data["^GSPC_Price"], label="SP500 Indexed Price", color="red")
       axs[1].set_title("Indexed Prices")
       axs[1].set_ylabel("Indexed Price")
       axs[1].legend()
       axs[1].grid(True)

       axs[2].plot(data["GS Volatility"], label="GS Volatility", color="blue")
       axs[2].plot(data["S&P500 Volatility"], label="S&P500 Volatility", color="red")
       axs[2].set_title("Conditional Volatility")
       axs[2].set_ylabel("Conditional Volatility")
       axs[2].legend()
       axs[2].grid(True)

       axs[3].plot(data["LRMES"], label="GS LRMES", color="blue")
       axs[3].set_title("LRMES - Conditional on a 40% Market Decline in Six Months")
       axs[3].set_xlabel("Date")
       axs[3].set_ylabel("LRMES")
       axs[3].legend()
       axs[3].grid(True)

       plt.tight_layout()
       plt.show()
       plt.savefig("./GS_LRMES.png")


