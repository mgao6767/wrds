######################
 GJR-GARCH(1,1) - DCC
######################

**************
 Introduction
**************

The Multivariate GARCH(1,1) model generalizes the univariate
:doc:`/algorithms/garch` framework to multiple time series, capturing
not only the conditional variances but also the conditional covariances
between the series. One common form is the Constant Conditional
Correlation (CCC) model proposed by Bollerslev (1990), discussed in
:doc:`/algorithms/garch-ccc`.

However, CCC model is limited by the assumption of a constant
correlation. Engle (2002) and Tse and Tsui (2002) address this by
proposing the **Dynamic Conditional Correlation (DCC)** model, which
allows for time-varying conditional correlation.

:doc:`/algorithms/garch-dcc` dicusses the GARCH-DCC, here, 
the :doc:`/algorithms/gjr-garch-dcc` model is discussed.

.. tip::

   Check Examples_ section for code guide and comparison to Stata and R.

Return equation
===============

The return equation for a :math:`N`-dimensional time series is:

.. math::
   :label: mv_return_eq

   \mathbf{r}_t = \boldsymbol{\mu} + \boldsymbol{\epsilon}_t

Here, :math:`\mathbf{r}_t` is a :math:`N \times 1` vector of returns,
and :math:`\boldsymbol{\mu}` is a :math:`N \times 1` vector of mean
returns. :math:`\boldsymbol{\epsilon}_t` is the :math:`N \times 1`
vector of shock terms.

Shock equation
==============

The shock term is modelled as:

.. math::
   :label: mv_shock_eq

   \boldsymbol{\epsilon}_t = \mathbf{H}_t^{1/2} \mathbf{z}_t

Here, :math:`\mathbf{H}_t` is a :math:`N \times N` conditional
covariance matrix, :math:`\mathbf{H}_t^{1/2}` is a :math:`N \times N`
positive definite matrix, and :math:`\mathbf{z}_t` is a :math:`N \times
1` vector of i.i.d. standard normal innovations.

.. math::
   :label: shock_dcc

   \mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I}_N)

Conditional covariance matrix
=============================

In the DCC-GJR-GARCH(1,1) model, the conditional covariance matrix
:math:`\mathbf{H}_t` is constructed as:

.. math::
   :label: mv_volatility_eq

   \mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t

where :math:`\mathbf{D}_t=\text{diag}(\mathbf{h}_t)^{1/2}`, and
:math:`\mathbf{h}_t` is a :math:`N \times 1` vector whose elements are
univariate GJR-GARCH(1,1) variances for each time series.
:math:`\mathbf{R}_t` is the :math:`N \times N` conditional correlation
matrix which is time-varying in DCC-GJR-GARCH.

.. caution::

   The log-likelihood function for the :math:`N`-dimensional
   multivariate GJR-GARCH-DCC model is:

   .. math::

      \ell = -\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \ln(|\mathbf{R}_t|)+ \mathbf{z}_t' \mathbf{R}_t^{-1} \mathbf{z}_t \right]

   The formulation of dynamic conditional covariance above implies that
   the time-varying :math:`\mathbf{R}_t` must be inverted at each time
   :math:`t`, which has to be positive definite as well. These
   constraints would make estimation extremely slow.

Engle (2002) achieves these constraints by modelling
:math:`\mathbf{R}_t` via a proxy process :math:`\mathbf{Q}_t`.
Specifically, the conditional correlation matrix :math:`\mathbf{R}_t`
can be obtained as:

.. math::
   :label: conditional_correlation_dcc

   \mathbf{R}_t = \text{diag}(\mathbf{Q}_t)^{-1/2} \mathbf{Q}_t \text{diag}(\mathbf{Q}_t)^{-1/2}

and the proxy process :math:`\mathbf{Q}_t` is

.. math::
   :label: dcc_equation

   \mathbf{Q}_t = (1 - a - b) \mathbf{\bar{Q}} + a (\mathbf{z}_{t-1}\mathbf{z}_{t-1}') + b \mathbf{Q}_{t-1}

Here, :math:`a` and :math:`b` are DCC parameters, and
:math:`\mathbf{\bar{Q}}` is the unconditional correlation matrix of
standardized residuals :math:`\mathbf{z}_t`.

Log-likelihood function
=======================

The log-likelihood function for the :math:`N`-dimensional multivariate
GJR-GARCH DCC model is:

.. math::
   :label: mv_log_likelihood

   \ell = -\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \ln(|\mathbf{R}_t|)+ \mathbf{z}_t' \mathbf{R}_t^{-1} \mathbf{z}_t \right]

where :math:`\mathbf{z}_t=\mathbf{D}_t^{-1}\mathbf{\epsilon}_t` is the
vector of standardized residuals. We can rewrite and decompose the
log-likelihood function by adding and subtracting
:math:`\mathbf{\epsilon}_t' \mathbf{D}_t^{-1} \mathbf{D}_t^{-1}
\mathbf{\epsilon}_t = \mathbf{z}_t'\mathbf{z}_t`,

.. math::
   :label: mv_log_likelihood_decomposed

   \ell = \underbrace{-\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \mathbf{\epsilon}_t' \mathbf{D}_t^{-1} \mathbf{D}_t^{-1}\mathbf{\epsilon}_t \right]}_{\ell_{V}(\Theta_1)\text{ volatility component}} \underbrace{-\frac{1}{2} \sum_{t=1}^T \left[ -\mathbf{z}_t'\mathbf{z}_t + \ln(|\mathbf{R}_t|)+ \mathbf{z}_t' \mathbf{R}_t^{-1} \mathbf{z}_t \right]}_{\ell_{C}(\Theta_1, \Theta_2)\text{ correlation component}}

This decomposition reveals an interesting fact. We can view the
loglikelihood as sum of two components.

#. :math:`\ell_{V}(\Theta_1)` is about the conditioal variances of the
   returns.
#. :math:`\ell_{C}(\Theta_1, \Theta_2)` is about the conditional
   correlation.

*****************************************
 Two-step quasi-maximum likelihood (QML)
*****************************************

The above loglikelihood decomposition suggests a twp-step approach in
MLE.

Specifically, given the assumption of multivariate normal, the
volatility component :math:`\ell_{V}(\Theta_1)` is the sum of individual
GJR-GARCH loglikelihood. It can be maximized by separately maximizing each
univariate model. So, we can separately estimate for each returns a
GJR-GARCH model via MLE, and add up the loglikelihoods. This is the first
step.

After the first step, we have the parameters
:math:`\Theta_1=(\mu,\omega,\alpha,\gamma,\beta)` for the GJR-GARCH models, and we
can then estimate the remaining parameters :math:`\Theta_2=(a, b)`.

****************
 Bivariate case
****************

The return equations for the two time series at time :math:`t` are:

.. math::
   :label: return_eqn_1

   r_{1t} = \mu_1 + \epsilon_{1t}

.. math::
   :label: return_eqn_2

   r_{2t} = \mu_2 + \epsilon_{2t}

Here, :math:`\epsilon_{1t} = \sqrt{h_{1t}} z_{1t}` and
:math:`\epsilon_{2t} = \sqrt{h_{2t}} z_{2t}`, with :math:`z_{1t}, z_{2t}
\sim N(0,1)`.

The conditional variances :math:`h_{1t}` and :math:`h_{2t}` are
specified as:

.. math::
   :label: h1

   h_{1t} = \omega_1 + \alpha_1 \epsilon_{1,t-1}^2 + \gamma_1 \epsilon_{1,t-1}^2 I_{1,t-1} + \beta_1 h_{1,t-1}

.. math::
   :label: h2

   h_{2t} = \omega_2 + \alpha_2 \epsilon_{2,t-1}^2 + \gamma_2 \epsilon_{2,t-1}^2 I_{2,t-1} + \beta_2 h_{2,t-1}

Here, :math:`I_{i,t-1} = 1` if :math:`\epsilon_{i,t-1} < 0` and 0 otherwise.

The dynamic conditional correlation :math:`\rho_t` is given by:

.. math::
   :label: rho_t

   \rho_t = \frac{q_{12t}}{\sqrt{q_{11t} q_{22t}}}

The Q process is updated as follows:

.. math::
   :label: q11

   q_{11t} = (1 - a - b) \overline{q}_{11} + a z_{1,t-1} z_{1,t-1} + b q_{11,t-1}

.. math::
   :label: q12

   q_{12t} = (1 - a - b) \overline{q}_{12} + a z_{1,t-1} z_{2,t-1} + b q_{12,t-1}

.. math::
   :label: q22

   q_{22t} = (1 - a - b) \overline{q}_{22} + a z_{2,t-1} z_{2,t-1} + b q_{22,t-1}

The log-likelihood function :math:`\ell` can be decomposed into two
components:

.. math::
   :label: log_likelihood_total

   \ell = \ell_{V}(\Theta_1) + \ell_{C}(\Theta_1, \Theta_2)

The first part, :math:`\ell_{V}(\Theta_1)`, is the sum of individual
GJR-GARCH log-likelihoods and is given by:

.. math::
   :label: ll_volatility

   \ell_{V}(\Theta_1) = -\frac{1}{2} \sum_{t=1}^T \left[ 2\ln(2\pi) + \ln(h_{1t}) + \ln(h_{2t}) + \frac{\epsilon_{1t}^2}{h_{1t}} + \frac{\epsilon_{2t}^2}{h_{2t}} \right]

The second part, :math:`\ell_{C}(\Theta_1, \Theta_2)`, focuses on the
correlation and is given by:

.. math::
   :label: ll_correlation

   \ell_{C}(\Theta_1, \Theta_2) = -\frac{1}{2} \sum_{t=1}^T \left[ -\left(z_{1t}^2 + z_{2t}^2\right) + \ln(1 - \rho_t^2) + \frac{z_{1t}^2 + z_{2t}^2 - 2\rho_t z_{1t} z_{2t}}{1 - \rho_t^2} \right]

Here,

-  :math:`z_{1t}` and :math:`z_{2t}` are the standardized residuals.

-  :math:`\rho_t` is the dynamic conditional correlation, derived from
   :math:`q_{11t}`, :math:`q_{12t}`, and :math:`q_{22t}`.

-  :math:`\Theta_1` includes the parameters for the individual GJR-GARCH
   models: :math:`\mu_1, \omega_1, \alpha_1, \gamma_1, \beta_1, \mu_2, \omega_2,
   \alpha_2, \gamma_2, \beta_2`.

-  :math:`\Theta_2` includes the parameters for the DCC model:
   :math:`\alpha, \beta`.

**********************
 Esimation techniques
**********************

My implementation of :class:`frds.algorithms.GJRGARCHModel_DCC` fits the
GJR-GARCH-DCC model by a two-step quasi-maximum likelihood (QML) method.

Step 1. Use :class:`frds.algorithms.GJRGARCHModel` to estimate the
:doc:`/algorithms/gjr-garch` model for each of the returns. This step yields
the estimates :math:`\hat{\Theta}_1`, including parameters for the
individual GJR-GARCH models: :math:`\mu_1, \omega_1, \alpha_1, \beta_1,
\mu_2, \omega_2, \alpha_2, \beta_2`. We obtain also the maximized
log-likelihood :math:`\ell(\hat{\Theta}_1)`.

Step 2. Use the estimated parameters from Step 1 to maximize
:math:`\ell_{C}(\hat{\Theta}_1, \Theta_2)` with respect to
:math:`\Theta_2=(a,b)`. A grid search is performed to find the starting values 
of :math:`(a,b)` based on loglikelihood.

************
 References
************

-  `Engle, R. F. (1982) <https://doi.org/10.2307/1912773>`_,
   "Autoregressive Conditional Heteroskedasticity with Estimates of the
   Variance of United Kingdom Inflation." *Econometrica*, 50(4),
   987-1007.

-  `Bollerslev, T. (1990) <https://doi.org/10.2307/2109358>`_,
   "Modelling the Coherence in Short-Run Nominal Exchange Rates: A
   Multivariate Generalized ARCH Model." *Review of Economics and
   Statistics*, 72(3), 498-505.

-  `Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993) <https://doi.org/10.1111/j.1540-6261.1993.tb05128.x>`_, "On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *The Journal of Finance*, 48(5), 1779-1801.

-  `Engle, R. (2002) <https://www.jstor.org/stable/1392121>`_, "Dynamic
   Conditional Correlation: A Simple Class of Multivariate Generalized
   Autoregressive Conditional Heteroskedasticity Models." *Journal of
   Business & Economic Statistics*, 20(3), 339-350.

-  `Tse, Y.K. and Tsui, A.K.C. (2002) <https://www.jstor.org/stable/1392122>`_,
   "A Multivariate Generalized Autoregressive Conditional Heteroskedasticity Model with Time-Varying Correlations." 
   *Journal of Business & Economic Statistics*, 20(3), 351-362.

*****
 API
*****

.. autoclass:: frds.algorithms.GJRGARCHModel_DCC
   :exclude-members: Parameters

.. autoclass:: frds.algorithms.GJRGARCHModel_DCC.Parameters
   :exclude-members: __init__
   :no-undoc-members:

**********
 Examples
**********

Let's import the dataset.

>>> import pandas as pd
>>> data_url = "https://www.stata-press.com/data/r18/stocks.dta"
>>> df = pd.read_stata(data_url, convert_dates=["date"])

Scale returns to percentage returns for better optimization results

>>> returns1 = df["toyota"].to_numpy() * 100
>>> returns2 = df["nissan"].to_numpy() * 100

frds
======

Use :class:`frds.algorithms.GJRGARCHModel_DCC` to estimate a GJR-GARCH(1,1)-DCC. 

>>> from frds.algorithms import GJRGARCHModel_DCC
>>> model_dcc = GJRGARCHModel_DCC(returns1, returns2)
>>> res = model_dcc.fit()
>>> from pprint import pprint
>>> pprint(res)
Parameters(mu1=0.03425396110878375,
           omega1=0.02870349714933671,
           alpha1=0.0629604836797677,
           gamma1=0.012013473922807561,
           beta1=0.9217503095555597,
           mu2=0.010528449295629098,
           omega2=0.05512898468355955,
           alpha2=0.07700974411970742,
           gamma2=0.021814015760057957,
           beta2=0.9013499076166999,
           a=0.04192697529292123,
           b=0.8978328716537962,
           loglikelihood=-7259.03519837521)

These results are slighly different from the ones obtained in R, but with 
marginally better loglikelihood overall.

R
===

In R, we can estimate the DCC-GRJGARCH(1,1) as

.. code-block:: R

   library(rmgarch)
   library(haven)
   stocks <- read_dta("https://www.stata-press.com/data/r18/stocks.dta")
   data <- data.frame(toyota=stocks$toyota*100, nissan=stocks$nissan*100)
   uspec <- multispec(replicate(2, ugarchspec(mean.model=list(armaOrder=c(0, 0)),
                                             variance.model=list(model="gjrGARCH", garchOrder=c(1, 1)))))
   dccspec <- dccspec(uspec=uspec, dccOrder=c(1, 1), distribution="mvnorm")
   dcc_fit <- dccfit(dccspec, data=data)
   dcc_fit

The results are:

.. code-block:: R

   *---------------------------------*
   *          DCC GARCH Fit          *
   *---------------------------------*

   Distribution         :  mvnorm
   Model                :  DCC(1,1)
   No. Parameters       :  13
   [VAR GARCH DCC UncQ] : [0+10+2+1]
   No. Series           :  2
   No. Obs.             :  2015
   Log-Likelihood       :  -7260.429
   Av.Log-Likelihood    :  -3.6 

   Optimal Parameters
   -----------------------------------
                  Estimate  Std. Error  t value Pr(>|t|)
   [toyota].mu      0.035250    0.031485  1.11959 0.262887
   [toyota].omega   0.029322    0.015296  1.91701 0.055237
   [toyota].alpha1  0.064252    0.015871  4.04852 0.000052
   [toyota].beta1   0.920387    0.017760 51.82272 0.000000
   [toyota].gamma1  0.011615    0.017344  0.66967 0.503066
   [nissan].mu      0.009901    0.036277  0.27292 0.784911
   [nissan].omega   0.057227    0.029777  1.92185 0.054625
   [nissan].alpha1  0.079891    0.035086  2.27702 0.022785
   [nissan].beta1   0.898218    0.031844 28.20660 0.000000
   [nissan].gamma1  0.021556    0.023028  0.93610 0.349221
   [Joint]dcca1     0.042226    0.010225  4.12983 0.000036
   [Joint]dccb1     0.897648    0.031025 28.93326 0.000000

These are more comparable to the results of :class:`frds.algorithms.GJRGARCHModel_DCC`,
because the package ``rmgarch`` also uses a 2-stage approach.

Stata
=======

.. note::

   Stata does not support multivariate GJR-GARCH with DCC.

   The ``tarch`` option is not allowed for ``-mgarch-`` command.

