=========================
Contingent Claim Analysis
=========================

Introduction
============

The difference between put price and CDS price as a measure of firm's 
contribution to systemic risk based on `Gary and Jobst (2010) <https://ideas.repec.org/p/imf/imfwpa/2013-054.html>`_. 

References
==========

* `Gary and Jobst (2010) <https://ideas.repec.org/p/imf/imfwpa/2013-054.html>`_, Systemic contingent claims analysis: Estimating market-implied systemic risk, *IMF Working Papers*, No 13/54.

* `Bisias, Flood, Lo, and Valavanis (2012)  <https://doi.org/10.1146/annurev-financial-110311-101754>`_, A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

API
===

.. autoclass:: frds.measures.ContingentClaimAnalysis

Examples
========

>>> from frds.measures import ContingentClaimAnalysis
>>> cca = ContingentClaimAnalysis()
>>> cca.estimate(
...     equity=5,
...     volatility=1.2,
...     risk_free_rate=0.02,
...     default_barrier=10,
...     time_to_maturity=20,
...     cds_spread=1.5,
... )
(6.659378336338627, 3.3467523905471133)