=============
Option Prices
=============

Introduction
============

European option values from Black-Scholes model, allowing dividends.

.. math::
   :label: eq:blsprice

   \begin{align}
   C &= S \times e^{-q \times T} \times N(d1) - K \times e^{-r \times T} \times N(d2) \\\\
   P &= K \times e^{-r \times T} \times N(-d2) - S \times e^{-q \times T} \times N(-d1)
   \end{align}

of which,

.. math::
   :label: eq:d1d2
  
   \begin{align}
   d1 &= \frac{\ln\left(\frac{S}{K}\right) + \left(r - q + \frac{\sigma^2}{2}\right) \times T}{\sigma \sqrt{T}} \\\\
   d2 &= d1 - \sigma \sqrt{T}
   \end{align}


where,

- :math:`C` = Call option price
- :math:`P` = Put option price
- :math:`S` = Current stock price
- :math:`K` = Strike price
- :math:`T` = Time to expiration (in years)
- :math:`r` = Risk-free interest rate (annualized)
- :math:`q` = Dividend yield (annualized)
- :math:`N(\cdot)` = Cumulative distribution function of the standard normal distribution
- :math:`\sigma` = Volatility of the underlying asset (annualized)

References
==========

* `Black and Scholes (1972) <https://www.jstor.org/stable/2978484>`_, The Valuation of Option Contracts and a Test of Market Efficiency, *The Journal of Finance*, 27(2), 399–417.

API
===

.. autofunction:: frds.measures.blsprice

Examples
========

>>> from frds.measures import blsprice
>>> blsprice(0.67, 0.7, 0.01, 5.0, 0.33, 0.002)
(0.19003370474049647, 0.1925609132790535)
