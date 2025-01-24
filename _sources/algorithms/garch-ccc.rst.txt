================
GARCH(1,1) - CCC
================

Introduction
============

The Multivariate GARCH(1,1) model generalizes the univariate :doc:`/algorithms/garch` 
framework to multiple time series, capturing not only the conditional variances 
but also the conditional covariances between the series. One common form is the 
**Constant Conditional Correlation (CCC) model** proposed by Bollerslev (1990).

.. tip:: Check `Examples`_ section for code guide and comparison to Stata.

Return equation
---------------

The return equation for a :math:`N`-dimensional time series is:

.. math::
   :label: mv_return_eq

   \mathbf{r}_t = \boldsymbol{\mu} + \boldsymbol{\epsilon}_t

Here, :math:`\mathbf{r}_t` is a :math:`N \times 1` vector of returns, and :math:`\boldsymbol{\mu}` is a :math:`N \times 1` vector of mean returns. :math:`\boldsymbol{\epsilon}_t` is the :math:`N \times 1` vector of shock terms.

Shock equation
--------------

The shock term is modelled as:

.. math::
   :label: mv_shock_eq

   \boldsymbol{\epsilon}_t = \mathbf{H}_t^{1/2} \mathbf{z}_t

Here, :math:`\mathbf{H}_t` is a :math:`N \times N` conditional covariance matrix, 
:math:`\mathbf{H}_t^{1/2}` is a :math:`N \times N` positive definite matrix,
and :math:`\mathbf{z}_t` is a :math:`N \times 1` vector of standard normal innovations.

Conditional covariance matrix
-----------------------------

In the CCC-GARCH(1,1) model, the conditional covariance matrix :math:`\mathbf{H}_t` is constructed as:

.. math::
   :label: mv_volatility_eq

   \mathbf{H}_t = \mathbf{D}_t\mathbf{R}\mathbf{D}_t

where :math:`\mathbf{D}_t=\text{diag}(\mathbf{h}_t)^{1/2}`, 
and :math:`\mathbf{h}_t` is a :math:`N \times 1` vector whose elements are univariate GARCH(1,1) variances for each time series.
:math:`\mathbf{R}` is a positive definite constant conditional correlation matrix.

.. admonition:: A bivarite example
   :class: note

   In a bivariate GARCH(1,1) setting, we have two univariate GARCH(1,1) processes, one for each return series.
   Specifically, the GARCH(1,1) equations for the conditional variances :math:`h_{1t}` and :math:`h_{2t}` can be written as:

   .. math::
      :label: garch1_modified

      h_{1t} = \omega_1 + \alpha_1 \epsilon_{1,t-1}^2 + \beta_1 h_{1,t-1}

   .. math::
      :label: garch2_modified

      h_{2t} = \omega_2 + \alpha_2 \epsilon_{2,t-1}^2 + \beta_2 h_{2,t-1}

   where,

   - :math:`\epsilon_{1,t-1}` and :math:`\epsilon_{2,t-1}` are past shock terms from their respective time series. 
   - The parameters :math:`\omega_1, \alpha_1, \beta_1, \omega_2, \alpha_2, \beta_2` are to be estimated.

   With these individual variances, the conditional covariance matrix :math:`\mathbf{H}_t` is:

   .. math::
      :label: conditional_cov_matrix_modified

      \mathbf{H}_t = \begin{pmatrix}
      h_{1t} & \rho\sqrt{h_{1t} h_{2t}} \\\\
      \rho\sqrt{h_{1t} h_{2t}} & h_{2t}
      \end{pmatrix}

   Here, :math:`\rho` is the correlation between the two time series. 
   It is assumed to be constant over time in the CCC-GARCH framework.

   The constant correlation matrix :math:`\mathbf{R}` simplifies to a :math:`2 \times 2` matrix:

   .. math::
      :label: bivariate_constant_corr_matrix
      
      \mathbf{R} = 
      \begin{pmatrix}
         1      & \rho \\\\
         \rho & 1
      \end{pmatrix}

Log-likelihood function
-----------------------

The log-likelihood function for the :math:`N`-dimensional multivariate GARCH CCC model is:

.. math::
   :label: mv_log_likelihood

   \begin{align}
   \ell &= -\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + \ln(|\mathbf{H}_t|) + \mathbf{\epsilon}_t' \mathbf{H}_t^{-1} \mathbf{\epsilon}_t \right] \\\\
        &= -\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + \ln(|\mathbf{D}_t\mathbf{R}\mathbf{D}_t|) + \mathbf{\epsilon}_t' \mathbf{D}_t^{-1}\mathbf{R}^{-1}\mathbf{D}_t^{-1} \mathbf{\epsilon}_t \right] \\\\
        &= -\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \ln(|\mathbf{R}|)+ \mathbf{z}_t' \mathbf{R}^{-1} \mathbf{z}_t \right] 
   \end{align}

where :math:`\mathbf{z}_t=\mathbf{D}_t^{-1}\mathbf{\epsilon}_t` is the vector of standardized residuals. 

This function is maximized to estimate the model parameters.

.. admonition:: A bivariate example
   :class: note

   In the bivariate case, the log-likelihood function can be specifically written as a function of all parameters.

   The log-likelihood function :math:`\ell` for the bivariate case with all parameters :math:`\Theta = (\mu_1, \omega_1, \alpha_1, \beta_1, \mu_2, \omega_2, \alpha_2, \beta_2, \rho)` is:

   .. math::
      :label: log_likelihood_bivariate

      \ell(\Theta) = -\frac{1}{2} \sum_{t=1}^T \left[ 2\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \ln(|\mathbf{R}|)+ \mathbf{z}_t' \mathbf{R}^{-1} \mathbf{z}_t \right] 

   Here, :math:`\mathbf{z}_t'` is the transpose of the vector of standardized residuals :math:`\mathbf{z}_t`, 

   .. math::

      \mathbf{z}_t = \begin{pmatrix}
        z_{1,t} \\
        z_{2,t}
      \end{pmatrix}

   Further,

   .. math::

      \mathbf{D}_t = \begin{pmatrix}
      \sqrt{h_{1t}} & 0 \\\\
      0 & \sqrt{h_{2t}}
      \end{pmatrix}

   so the log-determinant of :math:`\mathbf{D}_t` is

   .. math::

      \ln(|\mathbf{D}_t|) = \frac{1}{2} \ln(h_{1t} h_{2t}) 

   The log-determinant of :math:`\mathbf{R}` is 

   .. math::

      \ln(|\mathbf{R}|) = \ln(1 - \rho^2)

   Inverse of :math:`\mathbf{R}` is 

   .. math::

      \mathbf{R}^{-1} = \frac{1}{1 - \rho^2} \begin{pmatrix}
      1 & -\rho \\
      -\rho & 1
      \end{pmatrix}

   Lastly, :math:`\mathbf{z}_t' \mathbf{R}^{-1} \mathbf{z}_t` is

   .. math::

      \mathbf{z}_t' \mathbf{R}^{-1} \mathbf{z}_t = \frac{1}{1 - \rho^2} \left[ z_{1t}^2 - 2\rho z_{1t} z_{2t} + z_{2t}^2 \right]


   Inserting all of these into :math:`\ell(\Theta)` in equation :math:numref:`log_likelihood_bivariate`:

   .. math::
      :label: log_likelihood

      \ell(\Theta) = -\frac{1}{2} \sum_{t=1}^T \left[ 2\ln(2\pi) + \ln(h_{1t} h_{2t} (1 - \rho^2)) + \frac{1}{1 - \rho^2} \left( z_{1t}^2 - 2\rho z_{1t} z_{2t} + z_{2t}^2 \right) \right]

Estimation techniques
=====================

My implementation of :class:`frds.algorithms.GARCHModel_CCC` fits the GARCH-CCC model 
by simultaneously estimating all parameters via maxmimizing the log-likelihood :math:numref:`log_likelihood`.

General steps are:

1. Use :class:`frds.algorithms.GARCHModel` to estimate the :doc:`/algorithms/garch` model for each of the returns.

2. Use the standardized residuals from the estimated GARCH models to compute correlation coefficient.

3. Use as starting vaues the estimated parameters from above in optimizing the loglikelihood function.


References
==========

- `Engle, R. F. (1982) <https://doi.org/10.2307/1912773>`_, "Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007.

- `Bollerslev, T. (1990) <https://doi.org/10.2307/2109358>`_, "Modelling the Coherence in Short-Run Nominal Exchange Rates: A Multivariate Generalized ARCH Model." *Review of Economics and Statistics*, 72(3), 498-505.
  
API
===

.. autoclass:: frds.algorithms.GARCHModel_CCC
   :exclude-members: Parameters

.. autoclass:: frds.algorithms.GARCHModel_CCC.Parameters
   :exclude-members: __init__
   :no-undoc-members:

Examples
========

Let's import the dataset.

>>> import pandas as pd
>>> data_url = "https://www.stata-press.com/data/r18/stocks.dta"
>>> df = pd.read_stata(data_url, convert_dates=["date"])

Scale returns to percentage returns for better optimization results

>>> returns1 = df["toyota"].to_numpy() * 100
>>> returns2 = df["nissan"].to_numpy() * 100

Use :class:`frds.algorithms.GARCHModel_CCC` to estimate a GARCH(1,1)-CCC. 

>>> from frds.algorithms import GARCHModel_CCC
>>> model_ccc = GARCHModel_CCC(returns1, returns2)
>>> res = model_ccc.fit()
>>> from pprint import pprint
>>> pprint(res)
Parameters(mu1=0.02745814255283541,
           omega1=0.03401400758840226,
           alpha1=0.06593379740524756,
           beta1=0.9219575443861723,
           mu2=0.009390068254041505,
           omega2=0.058694325049554734,
           alpha2=0.0830561828957614,
           beta2=0.9040961791372522,
           rho=0.6506770477876749,
           loglikelihood=-7281.321453218112)

These results are comparable to the ones obtained in Stata, and even marginally 
better based on log-likelihood. In Stata, we can estimate the same model as below:

.. code-block:: stata

    webuse stocks, clear
    replace toyota = toyota * 100
    replace nissan = nissan * 100
    mgarch ccc (toyota nissan = ), arch(1) garch(1)
   
The Stata results are:

.. code-block:: stata

    Constant conditional correlation MGARCH model

     Sample: 1 thru 2015                                      Number of obs = 2,015
     Distribution: Gaussian                                   Wald chi2(.)  =     .
     Log likelihood = -7282.961                               Prob > chi2   =     .

     -------------------------------------------------------------------------------------
                         | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
     --------------------+----------------------------------------------------------------
     toyota              |
                 _cons   |   .0277462   .0302805     0.92   0.360    -.0316024    .0870948
     --------------------+----------------------------------------------------------------
     ARCH_toyota         |
                    arch |
                    L1.  |   .0666384   .0101597     6.56   0.000     .0467257    .0865511
                         |
                  garch  |
                    L1.  |   .9210688   .0119214    77.26   0.000     .8977032    .9444343
                         |
                  _cons  |   .0344153   .0109208     3.15   0.002      .013011    .0558197
     --------------------+----------------------------------------------------------------
     nissan              |
                  _cons  |   .0079682   .0349351     0.23   0.820    -.0605034    .0764398
     --------------------+----------------------------------------------------------------
     ARCH_nissan         |
                    arch |
                    L1.  |   .0851778   .0132656     6.42   0.000     .0591778    .1111779
                         |
                  garch  |
                    L1.  |   .9016613   .0150494    59.91   0.000     .8721649    .9311577
                         |
                  _cons  |   .0603765   .0178318     3.39   0.001     .0254269    .0953262
     --------------------+----------------------------------------------------------------
     corr(toyota,nissan) |   .6512249   .0128548    50.66   0.000       .62603    .6764199
     -------------------------------------------------------------------------------------

See `Stata's reference manual <https://www.stata.com/manuals/ts.pdf>`_ for its
estimation techniques.
