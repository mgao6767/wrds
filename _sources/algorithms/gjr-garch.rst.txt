==============
GJR-GARCH(1,1)
==============

Introduction
============

The GJR-GARCH model extends the basic :doc:`/algorithms/garch`  by accounting for leverage effects, where bad news (negative returns) has a greater impact on volatility than good news.

.. tip:: Check `Examples`_ section for code guide and comparison to Stata.

Return equation
---------------

There are many ways to specify return dynamics. Here a constant mean model is used.

.. math::
   :label: return_eq

   r_t = \mu + \epsilon_t

Here, :math:`r_t` is the asset return at time :math:`t`, :math:`\mu` is the mean of the asset return, and :math:`\epsilon_t` is the shock term.

Shock equation
--------------

The shock term :math:`\epsilon_t` can be decomposed as:

.. math::
   :label: shock_eq

   \epsilon_t = \sigma_t z_t

where :math:`\sigma_t` is the conditional volatility and :math:`z_t \sim N(0, 1)` a standard normal noise term.

.. note::

   We can also assume that the noise term follows a different distribution, such as Student-t, and modify the likelihood function below accordingly.

Volatility equation
-------------------

The conditional volatility in GJR-GARCH(1,1) is given by:

.. math::
   :label: volatility_eq

   \sigma^2_t = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 I_{t-1} + \beta \sigma_{t-1}^2

Here, :math:`I_{t-1} = 1` if :math:`\epsilon_{t-1} < 0` and 0 otherwise. :math:`\omega, \alpha, \gamma, \beta` are parameters to be estimated. The term :math:`\gamma \epsilon_{t-1}^2 I_{t-1}` captures the leverage effect.

.. admonition:: The unconditional variance and persistence
   :class: note

   The unconditional variance, often denoted as :math:`\text{Var}(\epsilon_t)` or :math:`\sigma^2`, refers to the long-run average or steady-state variance of the return series. It is the variance one would expect the series to revert to over the long term, and it doesn't condition on any past information.

   For a GJR-GARCH(1,1) model to be stationary, the **persistence** must be less than 1 ( :math:`\alpha + \beta + \gamma/2 < 1` ). Given this condition, the unconditional variance :math:`\sigma^2` can be computed as follows:

   .. math::
      :label: unconditional_variance_formula

      \sigma^2 = \frac{\omega}{1 - (\alpha + \gamma / 2 + \beta)}

   Note that the division by 2 for :math:`\gamma` assumes that the leverage effect occurs about half the time, given that :math:`I_{t-1}` takes the value 1 if the return is negative and 0 otherwise.

Log-likelihood function
-----------------------

The log-likelihood function for the GJR-GARCH(1,1) :math:`\ell(\mu, \omega, \alpha, \beta)` is:

.. math::
   :label: log_likelihood

   \ell(\mu, \omega, \alpha, \gamma, \beta)= -\frac{1}{2} \sum_{t=1}^T \left[ \ln(2\pi) + \ln(\sigma_t^2) + \frac{(r_t - \mu)^2}{\sigma_t^2} \right]

The parameters :math:`\mu, \omega, \alpha, \gamma, \beta` can then be estimated by maximizing this log-likelihood function.

Estimation techniques
=====================

A few tips to improve the estimation and enhance its numerical stability. 
These are basically the same as in estimating :doc:`/algorithms/garch`. 

Initial value of conditional variance
-------------------------------------

Note that the conditional variance in a GJR-GARCH(1,1) model is :math:numref:`volatility_eq`:

.. math::

   \sigma^2_t = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 I_{t-1} + \beta \sigma_{t-1}^2

We need a good starting value :math:`\sigma_0^2` to begin with, which can be estimated via the **backcasting technique**. Once we have that :math:`\sigma^2_0` through backcasting, we can proceed to calculate the entire series of conditional variances using the standard GARCH recursion formula.

To backcast the initial variance, we can use the Exponential Weighted Moving Average (EWMA) method, setting :math:`\sigma^2_0` to the EWMA of the sample variance of the first :math:`n \leq T` returns:

.. math::

   \sigma^2_0 = \sum_{t=1}^{n} w_t \cdot r_t^2

where :math:`w_t` are the exponentially decaying weights and :math:`r_t` are residuals of returns, i.e., returns de-meaned by sample average. This :math:`\sigma^2_0` is then used to derive :math:`\sigma^2_1` the starting value for the conditional variance series.

Initial value of :math:`\omega`
-------------------------------

The starting value of :math:`\omega` is relatively straightforward. Note that the unconditional variance is given by :math:numref:`unconditional_variance_formula`: :math:`\sigma^2 = \frac{\omega}{1-\alpha-\gamma/2-\beta}`. Therefore, given a level of persistence (:math:`\alpha+\gamma/2+\beta`), we can set the initial guess of :math:`\omega` to be the sample variance times one minus persistence:

.. math::

   \omega = \hat{\sigma}^2 \cdot \left(1-\alpha-\gamma/2-\beta\right)

where we use the known sample variance of residuals :math:`\hat{\sigma}^2` as a guess for the unconditional variance :math:`\sigma^2`. However, we still need to find good starting values for :math:`\alpha`, :math:`\gamma` and :math:`\beta`.

Initial value of :math:`\alpha`, :math:`\gamma` and :math:`\beta`
-----------------------------------------------------------------

Finding good starting values for :math:`\alpha` :math:`\gamma` and :math:`\beta` can be done by a small grid search.

- First, we don't know ex ante the persistence level, so we need to vary the persistence level from some low values to some high values, e.g., from 0.1 to 0.98.
- Second, generally the :math:`\alpha` parameter is not too big, for example, ranging from 0.01 to 0.2. 
- Third, the leverage effect :math:`\gamma` is generally not big too. We can set it to be in the same range as :math:`\alpha`.

We can permute combinations of the persistence level, :math:`\alpha` and :math:`\gamma`, which naturally gives the corresponding :math:`\beta` and hence :math:`\omega`. The "optimal" set of initial values of :math:`\omega, \alpha, \gamma, \beta` is the one that gives the highest log-likelihood.

.. note::

   The initial value of :math:`\mu` is reasonably set to the sample mean return.

Variance bounds
---------------

Another issue is that we want to ensure that in the estimation, condition variance 
does not blow up to infinity or becomes zero. Hence, we need to 
construct bounds for conditional variances during the GJR-GARCH(1,1) parameter estimation process. 

To do this, we can calculate loose lower and upper bounds for each observation.
Specifically, we can use sample variance of the residuals to compute global lower and upper 
bounds. We then use EWMA to compute the conditional variance for each time point.
The EWMA variances are then adjusted to ensure they are within global bounds.
Lastly, we scale the adjusted EWMA variances to form the variance bounds at each
time.

During the estimation process, whenever we compute the conditional variances based 
on the prevailing model parameters, we ensure that they are adjusted to be reasonably
within the bounds at each time.

References
==========

- `Engle, R. F. (1982) <https://doi.org/10.2307/1912773>`_, "Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007.

- `Bollerslev, T. (1986) <https://doi.org/10.1016/0304-4076(86)90063-1>`_, "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.

-  `Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993) <https://doi.org/10.1111/j.1540-6261.1993.tb05128.x>`_, "On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *The Journal of Finance*, 48(5), 1779-1801.

- ``arch`` by `Kevin Sheppard, et al <https://doi.org/10.5281/zenodo.593254>`_. 

API
===

.. autoclass:: frds.algorithms.GJRGARCHModel
   :private-members:
   :inherited-members:
   :exclude-members: Parameters

.. autoclass:: frds.algorithms.GJRGARCHModel.Parameters
   :exclude-members: __init__
   :no-undoc-members:

Examples
========

Let's import the dataset.

>>> import pandas as pd
>>> data_url = "https://www.stata-press.com/data/r18/stocks.dta"
>>> df = pd.read_stata(data_url, convert_dates=["date"])

Scale returns to percentage returns for better optimization results

>>> returns = df["nissan"].to_numpy() * 100

Use :class:`frds.algorithms.GJRGARCHModel` to estimate a GJR-GARCH(1,1). 

>>> from frds.algorithms import GJRGARCHModel
>>> from pprint import pprint
>>> model = GJRGARCHModel(returns)
>>> res = model.fit()
>>> pprint(res)
Parameters(mu=0.010528449295629098,
           omega=0.05512898468355955,
           alpha=0.07700974411970742,
           gamma=0.021814015760057957,
           beta=0.9013499076166999,
           loglikelihood=-4085.741514140086)

These estimates are identical to the ones produced by `arch <https://pypi.org/project/arch/>`_.

>>> from arch import arch_model
>>> model = arch_model(returns, mean='Constant', vol='GARCH', p=1, o=1, q=1)
>>> model.fit(disp=False)
                   Constant Mean - GJR-GARCH Model Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.000
Mean Model:             Constant Mean   Adj. R-squared:                  0.000
Vol Model:                  GJR-GARCH   Log-Likelihood:               -4085.74
Distribution:                  Normal   AIC:                           8181.48
Method:            Maximum Likelihood   BIC:                           8209.52
                                        No. Observations:                 2015
Date:                Fri, Sep 29 2023   Df Residuals:                     2014
Time:                        09:59:37   Df Model:                            1
                                  Mean Model
=============================================================================
                 coef    std err          t      P>|t|       95.0% Conf. Int.
-----------------------------------------------------------------------------
mu             0.0105  3.632e-02      0.290      0.772 [-6.066e-02,8.171e-02]
                               Volatility Model
=============================================================================
                 coef    std err          t      P>|t|       95.0% Conf. Int.
-----------------------------------------------------------------------------
omega          0.0551  2.901e-02      1.900  5.743e-02   [-1.740e-03,  0.112]
alpha[1]       0.0770  3.428e-02      2.247  2.467e-02    [9.821e-03,  0.144]
gamma[1]       0.0218  2.214e-02      0.985      0.324 [-2.158e-02,6.522e-02]
beta[1]        0.9014  3.159e-02     28.532 4.682e-179      [  0.839,  0.963]
=============================================================================

Additionaly, in Stata, we can estimate the same model as below:

.. code-block:: stata

    webuse stocks, clear
    replace nissan = nissan * 100
    arch nissan, arch(1) tarch(1) garch(1) vce(robust)

.. admonition:: Important! :math:`\gamma` in Stata
   :class: warning

   In Stata, the estimate of :math:`\gamma` is the negative of the ones obtained
   in ``arch`` and :class:`frds.algorithms.GJRGARCHModel`!

   This is because in Stata, GJR-GARCH is specified via the ``tarch`` option, or 
   threshold ARCH. It defines the indicator :math:`I_{t-1}` in equation :math:numref:`volatility_eq` the opposite way, :math:`I_{t-1} = 1` if :math:`\epsilon_{t-1} > 0` and 0 otherwise.
  


It would produce very similar estimates. The discrepencies are likly due to the 
different optimization algorithms used. Based on loglikelihood, the estimates 
from ``arch`` and :class:`frds.algorithms.GJRGARCHModel` are marginally better.

Notes
=====

I did many tests, and in 99% cases :class:`frds.algorithms.GJRGARCHModel` performs
equally well with ``arch``, simply because it's adapted from ``arch``. 
In some rare cases when the return series does not behave well, though, 
the two would produce very different estimates despite having almost identical log-likelihood.
