=======
Z-score
=======

Introduction
============

A measure of bank insolvency risk, defined as:

.. math::

   \text{Z-score} = \frac{\text{ROA}+\text{CAR}}{\sigma_{\text{ROA}}}

where :math:`\text{ROA}` is the bank's ROA, :math:`\text{CAR}` is the bank's capital ratio, and :math:`\sigma_{\text{ROA}}`
is the standard deviation of bank ROA.

The rationale behind Z-score is simple. A bank is insolvent when its loss :math:`-\pi` exceeds equity :math:`E`, i.e., :math:`-\pi>E`.
The probability of insolvency is :math:`P(-\pi>E)`.

If bank assets is :math:`A`, then :math:`P(-\pi>E)=P(-\frac{\pi}{A}>\frac{E}{A})=P(-ROA>CAR)`.

Assuming profits are normally distributed, then scaling :math:`(\text{ROA}+\text{CAR})` by :math:`\sigma_{\text{ROA}}` yields an
estimate of the distance to insolvency.

A higher Z-score implies that larger shocks to profitability are required to cause the losses to exceed bank equity.

References
==========

- `Laeven and Levine (2009) <https://doi.org/10.1016/j.jfineco.2008.09.003>`_,
  Bank governance, regulation and risk taking, *Journal of Financial Economics*, 93, 2, 259-275.
- `Houston, Lin, Lin and Ma (2010) <https://doi.org/10.1016/j.jfineco.2010.02.008>`_,
  Creditor rights, information sharing, and bank risk taking, *Journal of Financial Economics*, 96, 3, 485-512.
- `Beck, De Jonghe, and Schepens (2013) <https://doi.org/10.1016/j.jfi.2012.07.001>`_,
  Bank competition and stability: cross-country heterogeneity, *Journal of Financial Intermediation*, 22, 2, 218-244.
- `Delis, Hasan, and Tsionas (2014) <https://doi.org/10.1016/j.jbankfin.2014.03.024>`_,
  The risk of financial intermediaries, *Journal of Banking & Finance*, 44, 1-12.
- `Fang, Hasan, and Marton (2014) <https://doi.org/10.1016/j.jbankfin.2013.11.003>`_,
  Institutional development and bank stability: Evidence from transition countries, *Journal of Banking & Finance*, 39, 160-176.

API
===

.. autofunction:: frds.measures.z_score

Examples
========

>>> from frds.measures import z_score
>>> import numpy as np
>>> roas = np.array([0.1,0.2,0.15,0.18,0.2])
>>> z_score.estimate(roa=0.2, capital_ratio=0.5, past_roas=roas)
18.549962900111296
