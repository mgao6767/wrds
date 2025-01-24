====================
Lerner Index (Banks)
====================

Introduction
============

The Lerner Index is used to measure the market power of banks. It is defined as 
the markup of price over marginal cost, relative to the price. Mathematically, the Lerner Index :math:`L` is given by:

.. math::
   :label: lerner-index

   L = \frac{P - MC}{P}

where:

- :math:`P` is the price, often calculated as total bank revenue over assets.
- :math:`MC` is the marginal cost, which is usually estimated using econometric techniques.

.. note::

   `Elzinga and Mills (2011) <https://doi.org/10.1257/aer.101.3.558>`_ 
   "The Lerner Index of Monopoly Power: Origins and Uses" is a good read.

The marginal cost (:math:`MC`) is often estimated using advanced econometric models, such as the translog cost function. The translog cost function allows for more flexibility in capturing the complexities of banking operations compared to simpler functional forms.

The translog cost function :math:`\ln C(Q, W)` is often used to model the cost :math:`C`, e.g., total operating expenses, as a function of output :math:`Q` and a vector of input prices :math:`W`. The translog cost function can be represented as:

.. math::
   :label: lerner-index-translog

   \begin{align}
   \ln C(Q, W) &= \beta_0 + \beta_1 \ln Q + \frac{1}{2} \beta_2 (\ln Q)^2 \\\\
   &+ \sum_{i=1}^{n} \gamma_i \ln W_i + \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \theta_{ij} \ln W_i \ln W_j \\\\
   &+ \sum_{i=1}^{n} \phi_i \ln Q \ln W_i
   \end{align}

.. tip::

   Please refer to my post `Translog Production and Cost Functions <https://mingze-gao.com/posts/translog-production-and-cost-functions/>`_
   for details.

   Simply put, the use of translog cost function as specified in :math:numref:`lerner-index-translog`
   allows for approximating potentially very complex cost function, hence complex
   produciton function via duality.

After estimating this cost function, the marginal cost :math:`MC` with respect to the output :math:`Q` can be obtained by differentiating the cost function :math:`C(Q, W)` with respect to :math:`Q`:

.. math::
   :label: lerner-index-mc

   MC = \frac{\partial C(Q, W)}{\partial Q} = \frac{C}{Q} \left( \beta_1 + \beta_2 \ln Q + \sum_{i=1}^{n} \phi_i \ln W_i \right)

The output :math:`Q` is typically the total assets. [1]_

.. [1] `Shaffer and Spierdijk (2020) <https://doi.org/10.1016/j.jbankfin.2020.105859>`_ 
       Table 1 lists recent papers' measures of :math:`Q`, most of which are total assets. 

In the banking literature (e.g., `Berger, Klapper, and Turk-Ariss (2009) <https://doi.org/10.1007/s10693-008-0050-7>`_),
it is common to have three input factors: labour, funds, and fixed capital, so that we have

#. :math:`W_1` is the input price of labour, measured as the ratios of personnel expenses to total assets
#. :math:`W_2` is the input price of funds, measured as the interest expenses to total deposits 
#. :math:`W_3` is the input price of fixed capital, measured as other operating and administrative expenses to total assets

Improvements and Adjustments
============================

The above specification allows only one output :math:`Q`. Some studies allow banks to be 
multi-product and hence have several outputs. Here I do not discuss them.
A suggested reading is `Koetter, Kolari, and Spierdijk (2012) <https://doi.org/10.1162/REST_a_00155>`_.

Adjusted Lerner Index
---------------------

Moreover, `Koetter, Kolari, and Spierdijk (2012) <https://doi.org/10.1162/REST_a_00155>`_
propose the **adjusted Lerner Index**.

.. math::
   :label: efficiency-adjusted-lerner-index

   \text{adjusted } L = \frac{\pi + TC - MC \times Q}{\pi + TC}

where:

- :math:`\pi` is the bank's profit before tax
- :math:`TC` is the bank's total operating costs 

About this adjusted Lerner Index, `Clerides, Delis, and Kokas (2015) <https://doi.org/10.1111/fmii.12030>`_ comment

  Koetter, Kolari and Spierdijk (2012) have argued that the conventional approach 
  of computing the Lerner index assumes both profit efficiency (optimal choice 
  of prices) and cost efficiency (optimal choice of inputs by firms). As a result,
  the estimated price-cost margins do not correctly measure the true extent of market power. 
  The argument reflects a distinction pointed out by Lerner (1934) himself, who states that 
  “for practical purposes we must read monopoly power not as *potential* monopoly, but as monopoly *in force*” 
  (p. 170; italics as in the original). In other words, the Lerner index measures 
  actual (exercised) market power, while Koetter, Kolari, and Spierdijk (2012) 
  are interested in measuring potential market power.

Alternative specification
-------------------------

Of course, one we choose a different functional form for the cost function and obtain margainal cost :math:`MC`
accordingly. For example, `Clerides, Delis, and Kokas (2015) <https://doi.org/10.1111/fmii.12030>`_ and  
`Deli, Delis, Hasan, and Liu (2019) <https://doi.org/10.1016/j.jbankfin.2019.01.016>`_ use 
a log-linear cost function model.

A log-linear cost function is fundamentally a simplification of translog cost function
in that if uses a first-order Taylor expansion instead of second-order Taylor expansion.
Refer to my post `Translog Production and Cost Functions <https://mingze-gao.com/posts/translog-production-and-cost-functions/>`_
for the derivation of cost function.

Estimation of Translog Cost Function
====================================

.. tip:: 
   
   See my post `Translog Cost Function Etimation <https://mingze-gao.com/posts/translog-cost-function-estimation/>`_
   for a detailed discussion.

The Translog cost function can be estimated using either Ordinary Least Squares 
(OLS) or Stochastic Frontier Analysis (SFA), depending on the assumptions about the error term.

In OLS, the Translog cost function is estimated by minimizing the sum of squared residuals. The error term is assumed to be normally distributed and captures random noise.

In SFA, the error term is decomposed into two parts: one capturing random noise and another capturing inefficiency.

- **OLS** is simpler but does not separate inefficiency from random noise.
- **SFA** is more complex but provides additional insights into inefficiency.

.. important::

   In this current implementation, I use OLS to estimate the translog cost 
   function with time fixed effects and standard error clustering by firm. 
   Support for a SFA estimation via MLE may be added in the future.

Another important note is that we need to impose the **homogeneity of degree 1**
constraint, which basically states that if all input prices are scaled by a scalar 
:math:`s`, the cost would scale by the same :math:`s`. 

This is achieved by normalizing the total cost :math:`C` and input prices by :math:`W_1` 
(we can choose :math:`W_2` or :math:`W_3` instead). After normalization, we will 
have scaled :math:`\tilde{C}=\frac{C}{W_1}`, :math:`\tilde{W}_2=\frac{W_2}{W_1}` 
and :math:`\tilde{W}_3=\frac{W_3}{W_1}`.

.. tip:: 
   
   Again, see `Translog Cost Function Etimation <https://mingze-gao.com/posts/translog-cost-function-estimation/>`_
   for a detailed discussion and proof.

Time fixed effect
-----------------

With three inputs and one output, the model to estimate is then

.. math::
   :label: lerner-index-translog-ols-model

   \begin{align}
   \ln \tilde{C}_{it} &= \beta_0 + \beta_1 \ln Q_{it} + \frac{1}{2} \beta_2 (\ln Q_{it})^2 \\\\
   &+ \gamma_2 \ln\tilde{W}_{2,it} + \gamma_3 \ln\tilde{W}_{3,it} + \phi_3 \ln Q_{it} \ln\tilde{W}_{2,it} + \phi_3 \ln Q_{it} \ln\tilde{W}_{3,it} + d_{t} + \varepsilon_{it}
   \end{align}

where :math:`d_t` is the time fixed effects.

The marginal cost for bank :math:`i` at time :math:`t` is then obtained as 

.. math::
   :label: lerner-index-mc-from-estimates

   MC = \frac{\partial C}{\partial Q} = \frac{C}{Q}\frac{\partial \ln C}{\partial \ln Q} = \frac{C}{Q}\frac{\partial (\ln \tilde{C} + \ln W_1) }{\partial \ln Q} = \frac{C}{Q}\frac{\partial \ln \tilde{C} }{\partial \ln Q}

so

.. math::
   :label: lerner-index-mc-final

   MC_{it} = \frac{C_{it}}{Q_{it}} \left( \hat{\beta}_1 + \hat{\beta}_2 \ln Q + \hat{\phi}_2 \ln \tilde{W}_2 + \hat{\phi}_3 \ln \tilde{W}_3\right)

This :math:`MC_{it}` is used to compute the Lerner Index.

Model :math:numref:`lerner-index-translog-ols-model` adds year fixed effects 
to :math:numref:`lerner-index-translog` and can be estimated
with robust standard errors by bank to capture the specificities of each firm
(e.g., `Berger, Klapper, and Turk-Ariss (2009) <https://doi.org/10.1007/s10693-008-0050-7>`_,
`Beck, De Jonghe, and Schepens (2013) <https://doi.org/10.1016/j.jfi.2012.07.001>`_, among others)

Trend
-----

Alternatively, we can incorporate trend instead, similar to  
`Koetter, Kolari, and Spierdijk (2012) <https://doi.org/10.1162/REST_a_00155>`_,

.. math::
   :label: lerner-index-translog-ols-model-trend

   \begin{align}
   \ln \tilde{C}_{it} &= \beta_0 + \beta_1 \ln Q_{it} + \frac{1}{2} \beta_2 (\ln Q_{it})^2 \\\\
   &+ \gamma_2 \ln\tilde{W}_{2,it} + \gamma_3 \ln\tilde{W}_{3,it} + \phi_3 \ln Q_{it} \ln\tilde{W}_{2,it} + \phi_3 \ln Q_{it} \ln\tilde{W}_{3,it} \\\\
   &+ \eta_0 trend + \eta_1 trend^2 + \eta_2 trend \ln Q_{it} + \omega_{2} trend \ln\tilde{W}_{2,it} + \omega_{3} trend \ln\tilde{W}_{3,it} + \varepsilon_{it}
   \end{align}

such that the marginal cost is

.. math::
   :label: lerner-index-mc-final-trend

   MC_{it} = \frac{C_{it}}{Q_{it}} \left( \hat{\beta}_1 + \hat{\beta}_2 \ln Q + \hat{\phi}_2 \ln \tilde{W}_2 + \hat{\phi}_3 \ln \tilde{W}_3 + \eta_2 trend\right)


References
==========

- `Beck, De Jonghe, and Schepens (2013) <https://doi.org/10.1016/j.jfi.2012.07.001>`_,
  Bank competition and stability: Cross-country heterogeneity,
  *Journal of Financial Intermediation*, 22 (2), 218–244.

- `Berger, Klapper, and Turk-Ariss (2009) <https://doi.org/10.1007/s10693-008-0050-7>`_,
  Bank Competition and Financial Stability,
  *Journal of Financial Services Research*, 35, 99–118.
   
- `Clerides, Delis, and Kokas (2015) <https://doi.org/10.1111/fmii.12030>`_,
  A New Data Set On Competition In National Banking Markets,
  *Financial Markets, Institutions & Instruments*, 24, 267–311.

- `Deli, Delis, Hasan, and Liu (2019) <https://doi.org/10.1016/j.jbankfin.2019.01.016>`_,
  Enforcement of banking regulation and the cost of borrowing,
  *Journal of Banking & Finance*, 101, 147–160.

- `Elzinga and Mills (2011) <https://doi.org/10.1257/aer.101.3.558>`_,
  The Lerner Index of Monopoly Power: Origins and Uses,
  *American Economic Review*, 101(3), 558–564.

- `Koetter, Kolari, and Spierdijk (2012) <https://doi.org/10.1162/REST_a_00155>`_,
  Enjoying the Quiet Life under Deregulation? Evidence from Adjusted Lerner Indices for U.S. Banks,
  *Review of Economics and Statistics*, 94 (2), 462–480.

- `Shaffer and Spierdijk (2020) <https://doi.org/10.1016/j.jbankfin.2020.105859>`_,
  Measuring multi-product banks' market power using the Lerner index,
  *Journal of Banking & Finance*, Volume 117, August 2020, 105859.

API
===

.. note:: To be implemented.

.. autoclass:: frds.measures.LernerIndex

Examples
========
