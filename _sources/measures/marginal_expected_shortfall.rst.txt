===========================
Marginal Expected Shortfall
===========================

Introduction
============

The firm's average return during the 5% worst days for the market.

MES measures how exposed a firm is to aggregate tail shocks and, interestingly, 
together with leverage, it has a significant explanatory power for which firms 
contribute to a potential crisis as noted by 
`Acharya, Pedersen, Philippon, and Richardson (2017) <https://doi.org/10.1093/rfs/hhw088>`_.

It is used to construct the :doc:`/measures/systemic_expected_shortfall`.

References
==========

- `Acharya, Pedersen, Philippon, and Richardson (2017) <https://doi.org/10.1093/rfs/hhw088>`_,
  Measuring systemic risk, *The Review of Financial Studies*, 30, (1), 2-47.
- `Bisias, Flood, Lo, and Valavanis (2012) <https://doi.org/10.1146/annurev-financial-110311-101754>`_,
  A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

API
===

.. autoclass:: frds.measures.MarginalExpectedShortfall

Examples
========

>>> from numpy.random import RandomState
>>> from frds.measures import MarginalExpectedShortfall

Let's simulate some returns for the firm and the market.

>>> rng = RandomState(0)
>>> firm_returns = rng.normal(0,1,100)
>>> mkt_returns = rng.normal(0,1,100)

Compute the MES.

>>> mes = MarginalExpectedShortfall(firm_returns, mkt_returns)
>>> mes.estimate()
0.13494025343324562

