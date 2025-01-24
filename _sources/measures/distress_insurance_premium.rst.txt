==========================
Distress Insurance Premium
==========================

Introduction
============

A systemic risk metric by |HZZ2009| which represents a hypothetical
insurance premium against a systemic financial distress, defined as total
losses that exceed a given threshold, say 15%, of total bank liabilities.

.. _`Huang, Zhou, and Zhu (2009)`: https://doi.org/10.1016/j.jbankfin.2009.05.017

.. |HZZ2009| replace:: `Huang, Zhou, and Zhu (2009)`_

The methodology is general and can apply to any pre-selected group of firms with
publicly tradable equity and CDS contracts. Each institutions marginal
contribution to systemic risk is a function of its size, probability of default,
and asset correlation. The last two components need to be estimated from market
data.

The general steps are:

1. Use simulated asset returns from a joint normal distribution (using the correlations) to compute the distribution of joint defaults.
2. The loss-given-default (LGD) is assumed to follow a symmetric triangular distribution with a mean of 0.55 and in the range of :math:`[0.1,1]`.

.. note:: The mean LGD of 0.55 is taken down from the Basel II IRB formula.

3. Compute the probability of losses and the expected losses from the simulations.

References
==========

* |HZZ2009|, A framework for assessing the systemic risk of major financial institutions, *Journal of Banking & Finance*, 33(11), 2036-2049.

* `Bisias, Flood, Lo, and Valavanis (2012)  <https://doi.org/10.1146/annurev-financial-110311-101754>`_, A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

API
===

.. autoclass:: frds.measures.DistressInsurancePremium
    :members:

Examples
========

>>> import numpy as np
>>> from frds.measures import DistressInsurancePremium
>>> # hypothetical implied default probabilities of 6 banks
>>> default_probabilities = np.array([0.02, 0.10, 0.03, 0.20, 0.50, 0.15])
>>> # Hypothetical correlations of the banks' asset returns.
>>> correlations = np.array(
...     [
...         [ 1.000, -0.126, -0.637, 0.174,  0.469,  0.283],
...         [-0.126,  1.000,  0.294, 0.674,  0.150,  0.053],
...         [-0.637,  0.294,  1.000, 0.073, -0.658, -0.085],
...         [ 0.174,  0.674,  0.073, 1.000,  0.248,  0.508],
...         [ 0.469,  0.150, -0.658, 0.248,  1.000, -0.370],
...         [ 0.283,  0.053, -0.085, 0.508, -0.370,  1.000],
...     ]
... )
>>> dip = DistressInsurancePremium(default_probabilities, correlations)
>>> dip.estimate()
0.2865733550799999