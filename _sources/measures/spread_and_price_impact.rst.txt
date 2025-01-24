#########################
 Spread and Price Impact
#########################

***************
 Quoted Spread
***************

The percentage quoted spread is

.. math::
   :label: quoted-spread

   \text{Q-spread}=\frac{Ask_{it}-Bid_{it}}{m_{it}} \times 100

where :math:`m_{it}=(Ask_{it}+Bid_{it})/2` is the bid-ask midpoint at
time :math:`t`.

Alternatively, the log spread is

.. math::
   :label: quoted-spread-logj

   \text{Q-spread}=\ln Ask_{it}- \ln Bid_{it}

******************
 Effective Spread
******************

The effective spread measures a round trip cost for liquidity trader
selling the stock immediately after the purchase, so it is defined as
two times the absolute difference between the natural logarithms of the
transaction price and the quoted midpoint at the trading time.

The percentage effective spread is

.. math::
   :label: effective-spread

   \text{E-spread}=\frac{2\times|P_{it}-m_{it}|}{m_{it}} \times 100

Alternatively, the log spread is

.. math::
   :label: effective-spread-log

   \text{E-spread (log)}=2\times|\ln P_{it}-\ln m_{it}|

Trade directions can be included into the calculation of effective
spread. Specifically, if :math:`q_{it}=1` means a buy and
:math:`q_{it}=-1` means a sell, then we have

.. math::
   :label: effective-spread-with-direction

   \text{E-spread}=\frac{2 q_{it} (P_{it}-m_{it})}{m_{it}} \times 100

.. math::
   :label: effective-spread-log-with-direction

   \text{E-spread (log)}=2 q_{it} (\ln P_{it}-\ln m_{it})

.. note::

   JFE 2021: Bias in the effective bid-ask spread
   https://doi.org/10.1016/j.jfineco.2021.04.018

*****************
 Realized Spread
*****************

Realized spread is the temporary component of the effective spread. The
realized spread measures the profit or loss to a liquidity provider
assuming the she can close her position at the quoted bid-ask midpoint
sometime after the trade. Typically, this time difference between
opening and closing the position is set to five minutes
(:math:`\tau=5\text{min}`).

The percentage realized spread is

.. math::
   :label: realized-spread

   \text{R-spread}=\frac{2\times|P_{it}-m_{it+\tau}|}{m_{it}} \times 100

Alternatively, the log spread is

.. math::
   :label: realized-spread-log

   \text{R-spread (log)}=2\times|\ln P_{it}-\ln m_{it+\tau}|

It is also common to include trade direction into the calculation of
realized spread. Specifically, if :math:`q_{it}=1` means a buy and
:math:`q_{it}=-1` means a sell, then we have

.. math::
   :label: realized-spread-with-direction

   \text{R-spread}=\frac{2 q_{it} (P_{it}-m_{it+\tau})}{m_{it}} \times 100

.. math::
   :label: realized-spread-log-with-direction

   \text{R-spread (log)}=2 q_{it} (\ln P_{it}-\ln m_{it+\tau})

*********************
 Simple Price Impact
*********************

(Simple) Price impact is the permanent component of the effective
spread.

The percentage price impact is

.. math::
   :label: simple-price-impact

   \text{Simple Price Impact}=\frac{2\times|m_{it+\tau}-m_{it}|}{m_{it}} \times 100

Alternatively, the log version is

.. math::
   :label: simple-price-impact-log

   \text{Simple Price Impact (log)}=2\times|\ln m_{it+\tau}-\ln m_{it}|

To include trade direction into the calculation of realized spread,
where :math:`q_{it}=1` means a buy and :math:`q_{it}=-1` means a sell,
we have

.. math::
   :label: simple-price-impact-with-direction

   \text{Simple Price Impact}=\frac{2 q_{it} (m_{it+\tau}-m_{it})}{m_{it}} \times 100

.. math::
   :label: simple-price-impact-log-with-direction

   \text{Simple Price Impact (log)}=2 q_{it} (\ln m_{it+\tau}-\ln m_{it})

.. note::

   Basically, effective spread is the sum of realized spread and price
   impact.

*****
 API
*****

.. automodule:: frds.measures.spread

.. automodule:: frds.measures.price_impact

**********
 Examples
**********
