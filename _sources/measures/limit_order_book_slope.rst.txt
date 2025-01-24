########################
 Limit Order Book Slope
########################

**************
 Introduction
**************

Some various slope measures of the limit order book (LOB). A steeply
sloping order book indicates a lack of depth beyond the best bid or ask
quote, a sign of illiquidity.

.. note::

   The notations below largely follow the respective papers. Therefore,
   the same symbol may have different meanings.

   Certain measures are estimated by (typically hourly) snapshots or
   intervals, where quantities supplied at each level are aggregated. If
   so, the functions here should be applied on data by intervals.

Næs and Skjeltorp (2006)
=========================

`Næs and Skjeltorp (2006) <https://doi.org/10.1016/j.finmar.2006.04.001>`_ introduce the
slope measure of bid, ask and the LOB. They measure the slope at each
price level by the ratio of the increase in depth over the increase in
price, then average across all price levels.

.. math::
   :label: bid-slope-NæsSkjeltorp2006

   \text{Bid Slope}_{it} = \frac{1}{N^B} \left\{\frac{v^B_1}{|p^B_1/p_0 -1|} + \sum_{\tau=1}^{N^B-1} \frac{v^B_{\tau+1}/v^B_{\tau}-1}{|p^B_{\tau+1}/p^B_{\tau}-1|} \right\}

.. math::
   :label: ask-slope-NæsSkjeltorp2006

   \text{Ask Slope}_{it} = \frac{1}{N^A} \left\{\frac{v^A_1}{p^A_1/p_0 -1} + \sum_{\tau=1}^{N^A-1} \frac{v^A_{\tau+1}/v^A_{\tau}-1}{p^A_{\tau+1}/p^A_{\tau}-1} \right\}


where :math:`N^B` and :math:`N^A` are the total number of bid and ask
prices (tick levels) of stock :math:`i` (e.g., if we examine 5 levels of
LOB, :math:`N^B` and :math:`N^A` will be 5) and :math:`\tau` is the tick
level. The best bid (ask) price is denoted by :math:`p_1^B`
(:math:`p_1^A`), i.e. :math:`\tau=1`. :math:`p_0` is the bid-ask
midpoint. :math:`v_{\tau}^B` and :math:`v_{\tau}^A` denote the natural
logarithms of the *accumulated* volume at each tick level :math:`\tau`
on the bid and ask sides.

.. note::

   If we define :math:`V^B_\tau` as the total volume demanded at
   :math:`p^B_\tau`, then :math:`v_{\tau}^B = \ln \left(\sum_{j=1}^\tau
   V_j^B\right)`.

The LOB slope for stock :math:`i` at time :math:`t` is then computed as

.. math::
   :label: lob-slope-NæsSkjeltorp2006

   \text{LOB Slope}_{it} = \frac{\text{Bid Slope}_{it} + \text{Ask Slope}_{it}}{2}

Intuitively, the bid (ask) slope measures the percentage change in the
bid (ask) volume relative to the percentage change in the corresponding
bid (ask) price, which is averaged across all limit price levels in the
bid (ask) order book, and the LOB slope is an equally weighted average
of the bid and ask slopes.

Ghysels and Nguyen (2019)
=========================

`Ghysels and Nguyen (2019) <https://doi.org/10.3390/jrfm12040164>`_ take a
more direct approach. The slope is estimated from regressing the
percentage of cumulative depth on the price distance at each level.

.. math::
   :label: GhyselsNguyen2019

   QP_{it\tau} = \alpha_i + \beta_i \times |P_{it\tau} - m_{it}|  + \varepsilon_{it\tau}

where :math:`QP_{it\tau}` is the percentage of cumulative depth at level
:math:`\tau` at time :math:`t` and :math:`|P_{i\tau} - m_{it}|` is the
distance between the price at level :math:`\tau` and the best bid-ask
midpoint. :math:`\beta_i` is the estimated slope. The bid and ask slope
are estimated separately.

Because the cumulative depths are standardized by the total depth across
all :math:`K` levels, :math:`QP_{itK}` is always 100%. As such, the
slope measure :math:`\beta` depends only on how wide the :math:`K`-level
pricing grid is (the intercept :math:`\alpha` takes care of
:math:`QP_{it1}`). Accordingly, this slope measure captures the
tightness of prices in the book. A steeper slope indicates that limit
orders are priced closer to the market. Another way of interpreting the
slope is that it measures the elasticity of depth with respect to price.
A steeper slope implies that for one unit increase in price, there is a
greater increase in the quantity bid or offered, implying a greater
willingness of liquidity providers to facilitate trading demand on each
side of the market.

Hasbrouck and Seppi (2001)
==========================

Let :math:`B_k` and :math:`A_k` denote the per share bid and ask for
quote record :math:`k`, and let :math:`N_k^B` and :math:`N_k^A` denote
the respective number of shares posted at these quotes. Thus, a
prospective purchaser knows that, if hers is the first market buy order
to arrive, she can buy at least :math:`N_k^A` shares at the ask price
:math:`A_k`.

.. note::

   These slope measures below are for the whole LOB, not specific to bid
   or ask.

Quote Slope
-----------

.. math::
   :label: quote-slope-HS2001

   \text{Quote Slope}_k = \frac{A_k - B_k}{\log N^A_k + \log N^B_k}

Log Quote Slope
---------------

.. math::
   :label: log-quote-slope-HS2001

   \text{Log Quote Slope}_k = \frac{\log (A_k / B_k)}{\log N^A_k + \log N^B_k}

Della Vedova, Gao, Grant and Westerholm (working paper)
=======================================================

Slope is measured by the cumulative volume at level :math:`K` divided by
the distance from the :math:`K`-th level price to the bid-ask midpoint.

.. math::
   :label: bid-slope-DGGW

   \text{Bid Slope}_{it} = \frac{\sum_{x=1}^K \text{Bid Depth}_{itx}}{|\text{Bid Price}^K_{it} - m_{it}|}

.. math::
   :label: ask-slope-DGGW

   \text{Ask Slope}_{it} = \frac{\sum_{x=1}^K \text{Ask Depth}_{itx}}{\text{Ask Price}^K_{it} - m_{it}}

where :math:`\text{Depth}_{itx}` is the sum of the quantity of available
at bid/ask depth level :math:`x` in stock :math:`i` at time :math:`t`,
:math:`\text{Price}_{it}^K` is the bid/ask price at the :math:`K`-th
level, and :math:`m_{it}` is the bid-ask midpoint at time :math:`t`.

Bid-side Slope Difference
-------------------------

.. math::
   :label: bid-slope-diff-DGGW

   \text{Bid Slope Difference}_{it} = \frac{\text{Bid Depth}_{it5} - \text{Bid Depth}_{it3}}{|\text{Bid Price}_{it5} - \text{Bid Price}_{it3}|} - \frac{\text{Bid Depth}_{it3} - \text{Bid Depth}_{it1}}{|\text{Bid Price}_{it3} - \text{Bid Price}_{it1}|}

Ask-side Slope Difference
-------------------------

.. math::
   :label: ask-slope-diff-DGGW

   \text{Ask Slope Difference}_{it} = \frac{\text{Ask Depth}_{it5} - \text{Ask Depth}_{it3}}{|\text{Ask Price}_{it5} - \text{Ask Price}_{it3}|} - \frac{\text{Ask Depth}_{it3} - \text{Ask Depth}_{it1}}{|\text{Ask Price}_{it3} - \text{Ask Price}_{it1}|}


Scaled Depth Difference
-----------------------

In addition, the relative asymmetry of depth is also informative about
the relative demand and supply of the stock. Scaled Depth Difference
(SDD) is constructed to capture the relative asymmetry in the LOB at a
particular time up to a particular level :math:`K` (e.g., 5).

.. math::
   :label: scaled-depth-difference

   \text{SDD}_{i,t} = 2\times \frac{\sum_{x=1}^K \text{Ask Depth}_{i,t,x}-\sum_{x=1}^K \text{Bid Depth}_{i,t,x}}{\sum_{x=1}^K \text{Ask Depth}_{i,t,x}+\sum_{x=1}^K \text{Bid Depth}_{i,t,x}}

SDD is essentially the depth difference scaled by the average depth. A
value of SDD greater than zero indicates asymmetry in the direction of
the ask side of the book.

.. note::

   (TODO) Implementation of all other slope measures.

References
==========

-  `Næs and Skjeltorp (2006)
   <https://doi.org/10.1016/j.finmar.2006.04.001>`_, Order Book
   Characteristics and the Volume–Volatility Relation: Empirical
   Evidence from a Limit Order Market, *Journal of Financial Markets*
   9(4), 408–32.

-  `Valenzuela, Zer, Fryzlewicz and Rheinländer (2015)
   <https://doi.org/10.1016/j.finmar.2015.03.001>`_, Relative liquidity
   and future volatility, *Journal of Financial Markets*, 24, 25–48.

-  `Ghysels and Nguyen (2019) <https://doi.org/10.3390/jrfm12040164>`_,
   Price discovery of a speculative asset: Evidence from a bitcoin
   exchange, *Journal of Risk and Financial Management*, 12(4), 164.

-  `Duong and Kalev (2007) <http://dx.doi.org/10.2139/ssrn.1009549>`_,
   Order Book Slope and Price Volatility, *SSRN*.

API
====

.. automodule:: frds.measures.lob_slope