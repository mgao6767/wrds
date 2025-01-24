#######
 SRISK
#######

**************
 Introduction
**************

A conditional capital shortfall measure of systemic risk by `Brownlees
and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_.

Capital Shortfall
=================

Capital shortfall is a firm's required capital reserve minus the firm's
equity. Specifically, capital shortfall of a firm :math:`i` on day
:math:`t` is

.. math::
   :label: eq:capital_shortfall

   CS_{it} = kA_{it} - W_{it} = k(D_{it}+W_{it}) - W_{it}

where,

-  :math:`W_{it}` is the market value of equity
-  :math:`D_{it}` is the book value of debt
-  :math:`A_{it} = W_{it} + D_{it}` is the value of quasi assets
-  :math:`k` is the prudential capital fraction, set to 8%

A positive capital shortfall :math:`CS` means the firm is in distress,
i.e., the capital reserve required is larger than the firm's equity
value.

Systemic Event and SRISK
========================

A systemic event is a market decline below a threshold :math:`C` over a
time horizon :math:`h`.

If the multiperiod arithmetic market return between :math:`t+1` to
:math:`t+h` is :math:`R_{mt+1:t+h}`, then the systemic event is
:math:`\{R_{mt+1:t+h}<C\}`.

.. note::

   :math:`h=1` month and :math:`C=-10\%` are chosen in `Brownlees and
   Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_.

**SRISK** of a firm :math:`i` is its expected capital shortfall
conditional on a systemic event.

.. math::
   :label: eq:srisk

   SRISK_{it} = E_t(CS_{it+h} | R_{mt+1:t+h} < C)

The total amount of systemic risk in the financial system is measured as
the sum of all firm-level SRISK of the :math:`N` institutions in the
system with **positive** SRISK measures.

.. math::
   :label: eq:total_srisk

   SRISK_{t} = \sum_{i=1}^{N} SRISK_{it}

.. note::

   Institutions with negative SRISK are ignored. In a crisis it is
   unlikely that surplus capital will be easily mobilized through
   mergers or loans to support failing firms.

Computation of SRISK
====================

First, we expand :math:`CS_{it+h}`,

.. math::
   :label: eq:expand_srisk

   \begin{align*}
   SRISK_{it} &= E_t(CS_{it+h} | R_{mt+1:t+h} < C) \\\\
   &= k E_t(D_{it+h} | R_{mt+1:t+h} < C) + (1-k) E_t(W_{it+h} | R_{mt+1:t+h} < C)
   \end{align*}

If debt cannot be renegotiated in case of systemic event,

.. math::
   :label: eq:expected_debt

   E_t(D_{it+h} | R_{mt+1:t+h} < C)=D_{it}

So we have,

.. math::
   :label: eq:final_srisk

   \begin{align*}
   SRISK_{it} &= k D_{it} + (1-k) W_{it} (1 - LRMES_{it}) \\\\
   &= W_{it} [k LVG_{it} + (1-k) LRMES_{it} - 1]
   \end{align*}

where,

-  :math:`LVG_{it}` is quasi leverage ratio
   :math:`LVG_{it}=(D_{it}+W_{it})/W_{it}`.
-  :math:`LRMES_{it}` is :doc:`/measures/long_run_mes`, which captures
   the expected firm return conditional on a systemic event.

.. important::

   The key step in computing SRISK is estimating the
   :doc:`/measures/long_run_mes`.

   :math:`LRMES_{it}` for firm :math:`i` at time :math:`t` is then
   defined as

   .. math::

      LRMES_{it} = -E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]

   Refer to :doc:`/measures/long_run_mes` for the steps of estimating
   LRMES using :doc:`/algorithms/gjr-garch-dcc`.

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

.. autoclass:: frds.measures.SRISK

**********
 Examples
**********

>>> from frds.datasets import StockReturns
>>> returns = StockReturns.stocks_us

:class:`frds.datasets.StockReturns.stocks_us` provides daily stock returns of a few U.S. 
stocks, including Google, Goldman Sachs, JPMorgan, and the S&P500 index, from 2010 to 2022.

>>> returns.head()
               GOOGL        GS       JPM     ^GSPC
Date                                              
2010-01-05 -0.004404  0.017680  0.019370  0.003116
2010-01-06 -0.025209 -0.010673  0.005494  0.000546
2010-01-07 -0.023280  0.019569  0.019809  0.004001
2010-01-08  0.013331 -0.018912 -0.002456  0.002882
2010-01-11 -0.001512 -0.015776 -0.003357  0.001747
>>> len(returns)
3271

Below is a visualization of the returns and indexed prices.

.. image:: /images/stocks_us.png

Let's estimate some SRISKs. I'll use the last 600 days as the training sample.

>>> gs = returns["GS"].to_numpy()[-600:]
>>> jpm = returns["JPM"].to_numpy()[-600:]
>>> sp500 = returns["^GSPC"].to_numpy()[-600:]

We can estimate the SRISK for Goldman Sachs, assuming it has a market value 
equity of 100 and debt value of 900.

>>> from frds.measures import SRISK
>>> srisk = SRISK(gs, sp500, W=100.0, D=900.0)
>>> srisk.estimate()
-11.032087743990681

Negative SRISK! So Goldman Sachs with the assumed equity/debt is safe. 
What if we define a "systemic event" to be a market decline of 5% instead,
and assume a even higher leverage?

>>> srisk = SRISK(gs, sp500, W=100.0, D=1500.0)
>>> srisk.estimate(lrmes_C=-0.05)
33.462929665773935

Well, in this extreme case where the bank has a equity to debt ratio of 1/15, 
and a systemic event defined as market decline of 5% over 22 days, the SRISK of
the bank is positive suggesting a capital shortfall.
