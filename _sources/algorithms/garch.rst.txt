================
GARCH(1,1)
================

Introduction
============

The GARCH(1,1) model is a commonly used model for capturing the time-varying volatility in financial time series data. The model can be defined as follows:

.. tip:: Check `Examples`_ section for code guide and comparison to Stata.

Return equation
---------------

There are many ways to specify return dynamics. Here a constant mean model is used.

.. math::
   :label: eq:mean_model

   r_t = \mu + \epsilon_t

where :math:`r_t` represents the return at time :math:`t`, and :math:`\mu` is the mean return.

Shock equation
--------------

.. math::
   :label: eq:shock

   \epsilon_t = \sigma_t \cdot z_t

In this equation, :math:`\epsilon_t` is the shock term, :math:`\sigma_t` is the conditional volatility, and :math:`z_t` is a white noise error term with zero mean and unit variance (:math:`z_t \sim N(0,1)`).

.. note::

   We can also assume that the noise term follows a different distribution, such as Student-t, and modify the likelihood function below accordingly.

Volatility equation
-------------------

.. math::
   :label: eq:volatility_model

   \sigma_t^2 = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2

Here :math:`\sigma_t^2` is the conditional variance at time :math:`t`, and :math:`\omega`, :math:`\alpha`, :math:`\beta` are parameters to be estimated. This equation captures how volatility evolves over time.

.. admonition:: The unconditional variance and persistence
   :class: note

   The unconditional variance, often denoted as :math:`\text{Var}(\epsilon_t)` or :math:`\sigma^2`, refers to the long-run average or steady-state variance of the return series. It is the variance one would expect the series to revert to over the long term, and it doesn't condition on any past information.

   For a GARCH(1,1) model to be stationary, the **persistence**, sum of :math:`\alpha` and :math:`\beta`, must be less than 1 ( :math:`\alpha + \beta < 1` ). Given this condition, the unconditional variance :math:`\sigma^2` can be computed as follows:

   .. math::
      :label: eq:unconditional_variance

      \sigma^2 = \frac{\omega}{1 - \alpha - \beta}

   In this formulation, :math:`\omega` is the constant or "base" level of volatility, while :math:`\alpha` and :math:`\beta` determine how shocks to returns and past volatility influence future volatility. The unconditional variance provides a long-run average level around which the conditional variance oscillates.

Log-likelihood function
-----------------------

The likelihood function for a GARCH(1,1) model is used for the estimation of parameters :math:`\mu`, :math:`\omega`, :math:`\alpha`, and :math:`\beta`. Given a time series of returns :math:`\{ r_1, r_2, \ldots, r_T \}`, the likelihood function :math:`L(\mu, \omega, \alpha, \beta)` can be written as:

.. math::
   :label: eq:likelihood_func

   L(\mu, \omega, \alpha, \beta) = \prod_{t=1}^{T} \frac{1}{\sqrt{2\pi \sigma_t^2}} \exp\left(-\frac{(r_t-\mu)^2}{2\sigma_t^2}\right)

Taking the natural logarithm of :math:`L`, we obtain the log-likelihood function :math:`\ell(\mu, \omega, \alpha, \beta)`:

.. math::
   :label: eq:loglikelihood_func

   \ell(\mu, \omega, \alpha, \beta) = -\frac{1}{2} \sum_{t=1}^{T} \left[\ln(2\pi)  + \ln(\sigma_t^2) + \frac{(r_t-\mu)^2}{\sigma_t^2} \right]

The parameters :math:`\mu, \omega, \alpha, \beta` can then be estimated by maximizing this log-likelihood function.

Estimation techniques
=====================

A few tips to improve the estimation and enhance its numerical stability.

Initial value of conditional variance
-------------------------------------

Note that the conditional variance in a GARCH(1,1) model is :math:numref:`eq:volatility_model`:

.. math::

   \sigma_t^2 = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2

We need a good starting value :math:`\sigma_0^2` to begin with, which can be estimated via the **backcasting technique**. Once we have that :math:`\sigma^2_0` through backcasting, we can proceed to calculate the entire series of conditional variances using the standard GARCH recursion formula.

To backcast the initial variance, we can use the Exponential Weighted Moving Average (EWMA) method, setting :math:`\sigma^2_0` to the EWMA of the sample variance of the first :math:`n \leq T` returns:

.. math::

   \sigma^2_0 = \sum_{t=1}^{n} w_t \cdot r_t^2

where :math:`w_t` are the exponentially decaying weights and :math:`r_t` are residuals of returns, i.e., returns de-meaned by sample average. This :math:`\sigma^2_0` is then used to derive :math:`\sigma^2_1` the starting value for the conditional variance series.

Initial value of :math:`\omega`
-------------------------------

The starting value of :math:`\omega` is relatively straightforward. Notice that earlier we have jotted down the unconditional variance :math:`\sigma^2 = \frac{\omega}{1-\alpha-\beta}`. Therefore, given a level of persistence (:math:`\alpha+\beta`), we can set the initial guess of :math:`\omega` to be the sample variance times one minus persistence:

.. math::

   \omega = \hat{\sigma}^2 \cdot (1-\alpha-\beta)

where we use the known sample variance of residuals :math:`\hat{\sigma}^2` as a guess for the unconditional variance :math:`\sigma^2`. However, we still need to find good starting values for :math:`\alpha` and :math:`\beta`.

Initial value of :math:`\alpha` and :math:`\beta`
-------------------------------------------------

Unfortunately, there is no better way to find good starting values for :math:`\alpha` and :math:`\beta` than a grid search. Luckily, this grid search can be relatively small.

- First, we don't know ex ante the persistence level, so we need to vary the persistence level from some low values to some high values, e.g., from 0.1 to 0.98.
- Second, generally the :math:`\alpha` parameter is not too big, for example, ranging from 0.01 to 0.2.

We can permute combinations of the persistence level and :math:`\alpha`, which naturally gives the corresponding :math:`\beta` and hence :math:`\omega`. The "optimal" set of initial values of :math:`\omega, \alpha, \beta` is the one that gives the highest log-likelihood.

.. note::

   The initial value of :math:`\mu` is reasonably set to the sample mean return.

Variance bounds
---------------

Another issue is that we want to ensure that in the estimation, condition variance 
does not blow up to infinity or becomes zero. Hence, we need to 
construct bounds for conditional variances during the GARCH(1,1) parameter estimation process. 

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

- ``arch`` by `Kevin Sheppard, et al <https://doi.org/10.5281/zenodo.593254>`_. 

API
===

.. autoclass:: frds.algorithms.GARCHModel
   :private-members:
   :exclude-members: Parameters

.. autoclass:: frds.algorithms.GARCHModel.Parameters
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

Use :class:`frds.algorithms.GARCHModel` to estimate a GARCH(1,1). 

>>> from frds.algorithms import GARCHModel
>>> from pprint import pprint
>>> model = GARCHModel(returns)
>>> res = model.fit()
>>> pprint(res)
Parameters(mu=0.019315543596552513,
           omega=0.05701047522984261,
           alpha=0.0904653253307871,
           beta=0.8983752570013462,
           loglikelihood=-4086.487358003049)

These estimates are identical to the ones produced by `arch <https://pypi.org/project/arch/>`_.

>>> from arch import arch_model
>>> model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1)
>>> model.fit(disp=False)
                     Constant Mean - GARCH Model Results                      
==============================================================================
Dep. Variable:                      y   R-squared:                       0.000
Mean Model:             Constant Mean   Adj. R-squared:                  0.000
Vol Model:                      GARCH   Log-Likelihood:               -4086.49
Distribution:                  Normal   AIC:                           8180.97
Method:            Maximum Likelihood   BIC:                           8203.41
                                        No. Observations:                 2015
Date:                Thu, Sep 28 2023   Df Residuals:                     2014
Time:                        13:04:30   Df Model:                            1
                                  Mean Model                                 
=============================================================================
                 coef    std err          t      P>|t|       95.0% Conf. Int.
-----------------------------------------------------------------------------
mu             0.0193  3.599e-02      0.536      0.592 [-5.124e-02,8.985e-02]
                             Volatility Model                             
==========================================================================
                 coef    std err          t      P>|t|    95.0% Conf. Int.
--------------------------------------------------------------------------
omega          0.0570  2.810e-02      2.029  4.245e-02 [1.943e-03,  0.112]
alpha[1]       0.0905  2.718e-02      3.328  8.744e-04 [3.719e-02,  0.144]
beta[1]        0.8984  2.929e-02     30.670 1.426e-206   [  0.841,  0.956]
==========================================================================

Additionaly, in Stata, we can estimate the same model as below:

.. code-block:: stata

    webuse stocks, clear
    replace nissan = nissan * 100
    arch nissan, arch(1) garch(1) vce(robust)

It would produce very similar estimates. The discrepencies are likly due to the 
different optimization algorithms used. Based on loglikelihood, the estimates 
from ``arch`` and :class:`frds.algorithms.GARCHModel` are marginally better.

Notes
=====

I did many tests, and in 99% cases :class:`frds.algorithms.GARCHModel` performs
equally well with ``arch``, simply because it's adapted from ``arch``. 
In some rare cases when the return series does not behave well, though, 
the two would produce very different estimates despite having almost identical log-likelihood.
