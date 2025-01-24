###############
 Kyle's Lambda
###############

**************
 Introduction
**************

`Kyle (1985) <https://doi.org/10.2307/1913210>`_ Lamdba can be
interpreted as the cost of demanding a certain amount of liquidity over
a given time period. Following `Hasbrouck (2009)
<https://doi.org/10.1111/j.1540-6261.2009.01469.x>`_ and `Goyenko,
Holden and Trzcinka (2009)
<https://doi.org/10.1016/j.jfineco.2008.06.002>`_, we can estimate
Kyle's Lambda for stock :math:`i` as the slope coefficient
:math:`\lambda_{i}` in the following regression model:

.. math::
   :label: kylelambda_regression

   r_{i,n} = \lambda_{i}\cdot S_{i,n} + \varepsilon_{i,t}

where for the :math:`n`-th five-minute period, :math:`r_{i,n}` is the
(percentage) stock return and :math:`S_{i,n}` is the signed square-root dollar
volume, i.e., :math:`S_{i,n}=\sum_k sign(v_{k,n}) \sqrt{|v_{k,n}|}`, and
:math:`v_{kn}` is the signed dollar volume of the :math:`k`-th trade in
the :math:`n`-th five-minute period.

************
 References
************

-  `Kyle (1985) <https://doi.org/10.2307/1913210>`_, Continuous Auctions
   and Insider Trading, *Econometrica*, 53(6), 1315–1335.

-  `Hasbrouck (2009)
   <https://doi.org/10.1111/j.1540-6261.2009.01469.x>`_, Trading Costs
   and Returns for U.S. Equities: Estimating Effective Costs From Daily
   Data, *The Journal of Finance*, 64(3), 1445–1477.

-  `Goyenko, Holden and Trzcinka (2009)
   <https://doi.org/10.1016/j.jfineco.2008.06.002>`_, Do Liquidity
   Measures Measure Liquidity, *Journal of Financial Economics*, 92(2),
   153–181.

*****
 API
*****

.. autofunction:: frds.measures.kyle_lambda
