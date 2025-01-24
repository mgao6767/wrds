#######################################
 Probability of Informed Trading (PIN)
#######################################

**************
 Introduction
**************

In the market microstructure literature, Easley et. al. (1996) proposed
a trading model that can decompose the bid-ask spread. This model
introduces "Probability of Informed Trading", or **PIN**, which serves
as a means of measuring the informational component in the spread.

********************************
 Original Easley et. al. (1996)
********************************

Assume that the buy and sell orders of informed and uninformed traders
follow independent Poisson processes, and the following tree diagram
describes the entire trading process:

.. image:: /images/theoretical-model-of-pin.jpg
   :align: center
   :alt: Theoretical model of PIN

-  On each trading day, there is a probability of :math:`P=\alpha` that
   new information will appear, and obviously a probability of
   :math:`P=(1-\alpha)` that there will be no new information.

-  The probability of new information being bearish is :math:`P=\delta`,
   and the probability of it being bullish is :math:`P=(1-\delta)`.

   -  If the news is bearish, the arrival rate of buy orders on that day
      is :math:`\varepsilon`, and the arrival rate of sell orders is
      :math:`(\varepsilon+\mu)`.

   -  If the news is bullish, the arrival rate of buy orders on that day
      is :math:`(\varepsilon+\mu)`, and the arrival rate of sell orders
      is :math:`\varepsilon`.

-  When there is no new information, the arrival rate of both buy and
   sell orders is :math:`\varepsilon`.

Trading Process
===============

Next, assume that the market maker is a Bayesian, that is, he will
update his understanding of the overall market status, especially
whether there is new information on that day, by observing trades and
trading rates. Suppose each trading day is independent,
:math:`P(t)=(P_n(t), P_b(t), P_g(t))` is the market maker's prior
probability perception, where :math:`n` represents no new information,
:math:`b` represents bearish bad news, and :math:`g` represents bullish
good news, so :math:`P(t)=(1-\alpha, \alpha\delta, \alpha(1-\delta))`.

Let :math:`S_t` be the event of a sell order arriving at time :math:`t`,
and :math:`B_t` be the event of a buy order arriving at time :math:`t`.
Also, let :math:`P(t|S_t)` be the updated probability perception of the
market maker after observing a sell order arriving at time :math:`t`
based on the existing information. Then, according to Bayes' theorem, if
there is no new information at time :math:`t` and the market maker
observes a sell order, the posterior probability :math:`P_n(t|S_t)`
should be:

.. math::
   :label: eq1

   P_n(t|S_t) = \frac{P_n(t)\varepsilon}{\varepsilon+P_b(t)\mu}

Similarly, if there is bearish information and the market maker observes
a sell order at time :math:`t`, the posterior probability
:math:`P_b(t|S_t)` should be:

.. math::
   :label: eq2

   P_b(t|S_t) = \frac{P_b(t)(\varepsilon+\mu)}{\varepsilon+P_b(t)\mu}

If there is bullish information and the market maker observes a sell
order at time :math:`t`, the posterior probability :math:`P_g(t|S_t)`
should be:

.. math::
   :label: eq3

   P_g(t|S_t) = \frac{P_g(t)\varepsilon}{\varepsilon+P_b(t)\mu}

Thus, the expected zero-profit bid price at time :math:`t` on day
:math:`i` should be the conditional expectation of the asset value based
on historical information and observing a sell order at this time, that
is,

.. math::
   :label: eq4

   b(t) = \frac{P_n(t)\varepsilon V^*_i+P_b(t)(\varepsilon+\mu)\underline{V}_i+P_g(t)\varepsilon\overline{V}_i}{\varepsilon+P_b(t)\mu}

Here, :math:`V_i` is the value of the asset at the end of day :math:`i`,
and let the asset value be :math:`\overline{V}_i` when there is positive
news, :math:`\underline{V}_i` when there is negative news, and
:math:`V^*_i` when there is no news, with :math:`\underline{V}_i < V^*_i
< \overline{V}_i`.

At this point, the ask price should be:

.. math::
   :label: eq5

   a(t) = \frac{P_n(t)\varepsilon V^*_i+P_b(t)\varepsilon\underline{V}_i+P_g(t)(\varepsilon+\mu)\overline{V}_i}{\varepsilon+P_g(t)\mu}

Let's associate these bid and ask prices with the expected asset value
at time :math:`t`. Considering that the conditional expectation of the
asset value at this time is:

.. math::
   :label: eq6

   E[V_i|t] = P_n(t)V^*_i+P_b(t)\underline{V}_i+P_g(t)\overline{V}_i

we can write the above :math:`b(t)` and :math:`a(t)` as:

.. math::
   :label: eq7

   b(t) = E[V_i|t] - \frac{\mu P_b(t)}{\varepsilon+\mu P_b(t)}(E[V_i|t]-\underline{V}_i)

.. math::
   :label: eq8

   a(t) = E[V_i|t] + \frac{\mu P_g(t)}{\varepsilon+\mu P_g(t)}(\overline{V}_i-E[V_i|t])

Thus, the bid-ask spread is :math:`a(t)-b(t)`, which is:

.. math::
   :label: eq9

   a(t)-b(t) = \frac{\mu P_g(t)}{\varepsilon+\mu P_g(t)}(\overline{V}_i-E[V_i|t]) + \frac{\mu P_b(t)}{\varepsilon+\mu P_b(t)}(E[V_i|t]-\underline{V}_i)

This indicates that the bid-ask spread at time :math:`t` is actually:

-  The probability of a buy order being an informed trade times the
   expected loss due to the informed buyer
-  The probability of a sell order being an informed trade times the
   expected loss due to the informed seller

Therefore, the probability that any trade at time :math:`t` is based on
asymmetric information from informed traders is the sum of these two
probabilities:

.. math::
   :label: eq10

   PIN(t) = \frac{\mu P_g(t)}{\varepsilon+\mu P_g(t)} + \frac{\mu P_b(t)}{\varepsilon+\mu P_b(t)} = \frac{\mu(1-P_n(t))}{\mu(1-P_n(t))+2\varepsilon}

If no information event occurs (:math:`P_n(t)=1`) or there are no
informed trades (:math:`\mu=0`), both PIN and the bid-ask spread should
be zero. If the probabilities of positive and negative news are equal,
i.e., :math:`\delta=1-\delta`, the bid-ask spread can be simplified to:

.. math::
   :label: eq11

   a(t)-b(t) = \frac{\alpha\mu}{\alpha\mu+2\varepsilon}[\overline{V}_i-\underline{V}_i]

And the PIN measure is simplified to:

.. math::
   :label: eq12

   PIN(t) = \frac{\alpha\mu}{\alpha\mu+2\varepsilon}

Model Estimation
================

The parameters :math:`\theta=(\alpha, \delta, \varepsilon, \mu)` are not
easy to estimate because we can only observe the arrival of buy and sell
orders. In this model, the daily buy and sell orders are assumed to
follow one of the three Poisson processes. Although we don't know which
process it is specifically, the overall idea is: more buy orders imply
potential good news, more sell orders imply potential bad news, and
overall buying and selling will decrease when there is no new
information. With this idea in mind, we can try to estimate
:math:`\theta` using the maximum likelihood estimation (MLE) method.

First, according to the trading model shown in the diagram, assume that
there is bad news on a certain day, then the arrival rate of sell orders
is :math:`(\mu+\varepsilon)`, which means both informed and uninformed
traders participate in selling. The arrival rate of buy orders is
:math:`\varepsilon`, that is, only uninformed traders will continue to
buy. Therefore, the probability of observing a sequence of trades with
:math:`B` buy orders and :math:`S` sell orders in a period of time is:

.. math::
   :label: eq13

   e^{-\varepsilon} \frac{\varepsilon^B}{B!} e^{-(\mu+\varepsilon)} \frac{(\mu+\varepsilon)^S}{S!}

If there is good news on a certain day, the probability of observing a
sequence of trades with :math:`B` buy orders and :math:`S` sell orders
in a period of time is:

.. math::
   :label: eq14

   e^{-\varepsilon} \frac{\varepsilon^B}{B!} e^{-\varepsilon} \frac{\varepsilon^S}{S!}

If there is no new information on a certain day, the probability of
observing a sequence of trades with :math:`B` buy orders and :math:`S`
sell orders in a period of time is:

.. math::
   :label: eq15

   e^{-(\mu+\varepsilon)} \frac{(\mu+\varepsilon)^B}{B!} e^{-\varepsilon} \frac{\varepsilon^S}{S!}

So, the probability of observing a total of :math:`B` buy orders and
:math:`S` sell orders on a trading day should be the weighted average of
the above three possibilities, and the weights here are the
probabilities of each possibility. Therefore, we can write out the
likelihood function:

.. math::
   :label: eq16

   \begin{align}
   L((B, S)| \theta)= &(1-\alpha)e^{-\varepsilon} \frac{\varepsilon^B}{B!} e^{-\varepsilon} \frac{\varepsilon^S}{S!} \\\\
   &+ \alpha\delta  e^{-\varepsilon} \frac{\varepsilon^B}{B!} e^{-(\mu+\varepsilon)} \frac{(\mu+\varepsilon)^S}{S!} \\\\
   &+ \alpha(1-\delta) e^{-(\mu+\varepsilon)} \frac{(\mu+\varepsilon)^B}{B!} e^{-\varepsilon} \frac{\varepsilon^S}{S!}
   \end{align}

Because days are independent, the likelihood of observing the data
:math:`M=(B_i,S_i)_{i=1}^N` over :math:`N` days is jus the product of
the daily likelihoods. Hence, the objective function of the maximum
likelihood function is:

.. math::
   :label: eq17

   L(M|\theta)=\prod_{i=1}^{N}L(\theta|(B_i, S_i))

.. note::

   The real challenge is that this function is filled with powers and
   factorials. Even if the time element is chosen very small, some
   highly liquid assets will still have hundreds of transactions within
   a few seconds. Therefore, either :math:`B!`, :math:`S!`, or
   :math:`(\mu+\varepsilon)^B` can cause overflow. So, further
   processing of the objective function here is extremely important.

   Notice that in equation :math:numref:`eq16`, we can extract a
   common factor from the three terms
   :math:`e^{-2\varepsilon}(\mu+\varepsilon)^{B+S}/(B!S!)`. Afterwards,
   substitute :math:`x\equiv \frac{\varepsilon}{\mu+\varepsilon}\in [0,
   1]` into it. The transformed likelihood function, after taking the
   logarithm, will be in the form:

   .. math::
      :label: eq18

      \begin{align}
      l((B, S)| \theta)=&\ln(L((B, S)| \theta)) \\\\
      =&-2\varepsilon+(B+S)\ln(\mu+\varepsilon)  \\\\
      &+\ln((1-\alpha)x^{B+S}+\alpha\delta e^{-\mu}x^B + \alpha(1-\delta)e^{-\mu}x^S) \\\\
      &-\ln(B!S!)
      \end{align}

   Now, since the last term :math:`\ln(B!S!)` does not affect the
   parameter estimation at all, it can be safely excluded. The remaining
   part can avoid overflow as the introduction of :math:`x\equiv
   \frac{\varepsilon}{\mu+\varepsilon}\in [0, 1]` prevents the overflow
   error caused by :math:`(\mu+\varepsilon)>1`.

*****************************
 Easley et. al. (2002) Model
*****************************

Easley et. al. (2002) additionally allow different arrival rates for
buys :math:`\varepsilon_b` and sells :math:`\varepsilon_s`.

.. image:: /images/theoretical-model-of-pin-2002.jpg
   :align: center
   :alt: Theoretical model of PIN

The PIN measure is:

.. math::
   :label: PIN

   PIN(t) = \frac{\alpha\mu}{\alpha\mu+\varepsilon_b+\varepsilon_s}

After similar steps of derivation, the likelihood function is given by

.. math::
   :label: likelihoodfunction2002

   \begin{align}
   L((B, S)| \theta)= &(1-\alpha)e^{-\varepsilon_b} \frac{\varepsilon_b^B}{B!} e^{-\varepsilon_s} \frac{\varepsilon_s^S}{S!} \\\\
   &+ \alpha\delta  e^{-\varepsilon_b} \frac{\varepsilon_b^B}{B!} e^{-(\mu+\varepsilon_s)} \frac{(\mu+\varepsilon_s)^S}{S!} \\\\
   &+ \alpha(1-\delta) e^{-(\mu+\varepsilon_b)} \frac{(\mu+\varepsilon_b)^B}{B!} e^{-\varepsilon_s} \frac{\varepsilon_s^S}{S!}
   \end{align}

where :math:`\theta=(\alpha, \delta, \varepsilon_b, \varepsilon_s,
\mu)`.

Model Estimation
================

Many methods have been proposed to better estimate the likelihood function 
:math:numref:`likelihoodfunction2002` via MLE. Here I discuss two of
them, specifically, the log likelihood function after dropping the
constant term.

Easley, Hvidkjaer, and O’Hara (2010)
--------------------------------------

.. math::
   :label: EHO2010

      \begin{align}
      l((B_i, S_i)_{i=1}^N | \theta) =&\sum_{i=1}^N \left[ -\varepsilon_b -\varepsilon_s +M_i(\ln x_b + \ln x_s) + B_i \ln(\mu+\varepsilon_b) + S_i \ln(\mu+\varepsilon_s)\right]  \\\\
      &+\sum_{i=1}^N\ln\left[\alpha(1-\delta)e^{-\mu}x_s^{S_i-M_i}x_b^{-M_i} + \alpha\delta e^{-\mu} x_b^{B_i-M_i}x_s^{-M_i}+ (1-\alpha)x_s^{S_i-M_i}x_b^{B_i-M_i} \right]
      \end{align}

where :math:`M_i=\min(B_i, S_i)+\max(B_i, S_i)/2`,
:math:`x_s=\varepsilon_s/(\mu+\varepsilon_s)` and
:math:`x_b=\varepsilon_b/(\mu+\varepsilon_b)`. The factoring of
:math:`x_b^{M_i}` and :math:`x_s^{M_i}` is done to increase the
computing efficiency and reduce truncation error.

Lin and Ke (2011)
-----------------

Lin and Ke (2011) point out that floating-point exception in computer
software narrows the set of feasible solutions that maximizes the above
factorized likelihood function, which in turn causes a downward bias in
the estimate of PIN. They recommend the following factorization of the
joint likelihood function.

.. math::
   :label: LK2011

      \begin{align}
      l((B_i, S_i)_{i=1}^N | \theta) =&\sum_{i=1}^N \left[ -\varepsilon_b -\varepsilon_s + B_i \ln(\mu+\varepsilon_b) + S_i \ln(\mu+\varepsilon_s) + e_{\max i}\right]  \\\\
      &+\sum_{i=1}^N\ln\left[\alpha(1-\delta)\text{exp}(e_{1i}-e_{\max i}) + \alpha\delta\text{exp}(e_{2i}-e_{\max i}) + (1-\alpha)\text{exp}(e_{3i}-e_{\max i}) \right]
      \end{align}

where

-  :math:`e_{1i}=-\mu-S_i\ln(1+\mu/\varepsilon_s)`
-  :math:`e_{2i}=-\mu-B_i\ln(1+\mu/\varepsilon_b)`
-  :math:`e_{3i}=-B_i\ln(1+\mu/\varepsilon_b)-S_i\ln(1+\mu/\varepsilon_s)`
-  :math:`e_{\max i}=\max (e_{1i},e_{2i},e_{3i})`

Yan and Zhang (2012) further propose the set of initial values to use in
MLE.

************
 References
************

-  `Easley, Kiefer, O'Hara, and Paperman (1996)
   <https://doi.org/10.1111/j.1540-6261.1996.tb04074.x>`_, Liquidity,
   Information, and Infrequently Traded Stocks, *The Journal of
   Finance*, 51, 1405-1436.

-  `Easley, Hvidkjaer, and O'Hara (2002)
   <https://doi.org/10.1111/1540-6261.00493>`_, Is Information Risk a
   Determinant of Asset Returns?, *The Journal of Finance*, 57,
   2185-2221.

-  `Easley, Hvidkjaer, and O’Hara (2010)
   <https://doi.org/10.1017/S0022109010000074>`_, Factoring information
   into returns, *Journal of Financial and Quantitative Analysis*, 45,
   293–309.

-  `Lin and Ke (2011) <https://doi.org/10.1016/j.finmar.2011.03.001>`_,
   A computing bias in estimating the probability of informed trading,
   *Journal of Financial Markets*, 14, 625–640.

-  `Yan and Zhang (2012)
   <https://doi.org/10.1016/j.jbankfin.2011.08.003>`_, An improved
   estimation method and empirical properties of the probability of
   informed trading, *Journal of Banking & Finance*, 36(2), 454-467.

*****
 API
*****

.. autoclass:: frds.measures.PIN
   :private-members:
   :exclude-members: Parameters

.. autoclass:: frds.measures.PIN.Parameters
   :exclude-members: __init__
   :no-undoc-members:


**********
 Examples
**********

Consider the following example buys and sells consistent with the Easley, 
Hvidjkaer, and O'Hara example on p. 2198 of
Easley, David, Soeren Hvidkjaer, and Maureen O'Hara,
2002, "Is Information Risk a Determinant of Asset Returns?", 
The Journal of Finance, 57 (5), pp. 2185-2221.     

The parameters in this example would be identified as 
:math:`\varepsilon_b=\varepsilon_s=40`, :math:`\mu=50`, :math:`\alpha=0.4`, 
and :math:`\delta=0.5`.

>>> from pprint import pprint
>>> import numpy as np
>>> from frds.measures import PIN
>>> B = np.array([90, 40, 40, 40, 40])
>>> S = np.array([40, 90, 40, 40, 40])
>>> res = PIN(B, S).estimate(method="LK2011")
>>> pprint(res)
Parameters(alpha=0.4000034024442184,
           delta=0.50000867584611,
           epsilon_b=39.9999707607252,
           epsilon_s=40.00012455827183,
           mu=49.99988692576548,
           loglikelihood=1485.6558133261797,
           pin=0.20000080849728463,
           method='LK2011')
>>> res = PIN(B, S).estimate(method="EHO2010")
>>> pprint(res)
Parameters(alpha=0.400000965289364,
           delta=0.4999993508809301,
           epsilon_b=40.00007364091631,
           epsilon_s=40.000051685750314,
           mu=49.99989185329409,
           loglikelihood=1485.6558133270914,
           pin=0.19999978939239277,
           method='EHO2010')
