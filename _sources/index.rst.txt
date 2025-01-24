================================
Financial Research Data Services
================================

|frds|, *Financial Research Data Services*, is a Python library to simplify the 
complexities often encountered in financial research. It provides a collection 
of ready-to-use methods for computing a wide array of measures in the literature. 

It is developed by Dr. `Mingze Gao <http://mingze-gao.com>`_ from the Macquarie 
University, initially started as as a personal project during his postdoctoral 
research fellowship at the University of Sydney.

|GitHub license| |PyPI Downloads| |Tests| |PyPI Version|

.. |GitHub license| image:: https://img.shields.io/github/license/mgao6767/frds?color=blue
   :target: https://github.com/mgao6767/frds/blob/master/LICENSE

.. |PyPI Downloads| image:: https://img.shields.io/pypi/dm/frds?label=PyPI%20downloads 
   :target: https://pypi.org/project/frds/
   
.. |Tests| image:: https://github.com/mgao6767/frds/actions/workflows/test.yml/badge.svg
   :target: https://github.com/mgao6767/frds/actions/workflows/test.yml

.. |PyPI Version| image:: https://badge.fury.io/py/frds.svg
   :target: https://badge.fury.io/py/frds

.. |frds| replace:: :code:`frds`

.. important:: 
   This project is under active development. 
   Breaking changes may be expected.

   If there's any issue (likely), please contact me at 
   `mingze.gao@mq.edu.au <mailto:mingze.gao@mq.edu.au>`_.

------------
Quick start
------------

|frds| is available on `PyPI <https://pypi.org/project/frds/>`_ and can be 
installed via ``pip``.

.. code-block:: bash

   pip install frds

The structure of |frds| is simple:

* :mod:`frds.algorithms` provides a collection of algorithms.
* :mod:`frds.measures` provides a collection of measures.
* :mod:`frds.datasets` provides example datasets.

---------
Read more
---------

.. toctree::
    :maxdepth: 2

    measures/index
    algorithms/index
    datasets/index

.. toctree::
    :hidden:
    :caption: Other
    :titlesonly:

    GitHub <https://github.com/mgao6767/frds/>
    Mingze Gao <https://mingze-gao.com>
   
--------
Examples
--------

Some simple examples.

Measure
-------

:class:`frds.measures.DistressInsurancePremium` estimates
:doc:`/measures/distress_insurance_premium`, a systemic risk measure of a 
hypothetical insurance premium against a systemic financial distress, which is 
defined as total losses that exceed a given threshold, e.g., 15%, of total bank 
liabilities.

.. _`Huang, Zhou, and Zhu (2009)`: https://doi.org/10.1016/j.jbankfin.2009.05.017

.. |HZZ2009| replace:: `Huang, Zhou, and Zhu (2009)`_

>>> import numpy as np
>>> from frds.measures import DistressInsurancePremium
>>> # hypothetical implied default probabilities of 6 banks
>>> default_probabilities = np.array([0.02, 0.10, 0.03, 0.20, 0.50, 0.15]) 
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

Algorithm
---------

Use :class:`frds.algorithms.GARCHModel` to estimate a :doc:`/algorithms/garch` model.
The results are as good as those obtained from other software or libraries.

>>> import pandas as pd
>>> from pprint import pprint
>>> from frds.algorithms import GARCHModel
>>> data_url = "https://www.stata-press.com/data/r18/stocks.dta"
>>> df = pd.read_stata(data_url, convert_dates=["date"])
>>> nissan = df["nissan"].to_numpy() * 100
>>> model = GARCHModel(nissan)
>>> res = model.fit()
>>> pprint(res)
Parameters(mu=0.019315543596552513,
           omega=0.05701047522984261,
           alpha=0.0904653253307871,
           beta=0.8983752570013462,
           loglikelihood=-4086.487358003049)

Use :class:`frds.algorithms.GARCHModel_CCC` to estimate a bivariate :doc:`/algorithms/garch-ccc` model.
The results are as good as those obtained in Stata, if not better (based on loglikelihood).

>>> from frds.algorithms import GARCHModel_CCC
>>> toyota = df["toyota"].to_numpy() * 100
>>> model_ccc = GARCHModel_CCC(toyota, nissan)
>>> res = model_ccc.fit()
>>> pprint(res)
Parameters(mu1=0.02745814255283541,
           omega1=0.03401400758840226,
           alpha1=0.06593379740524756,
           beta1=0.9219575443861723,
           mu2=0.009390068254041505,
           omega2=0.058694325049554734,
           alpha2=0.0830561828957614,
           beta2=0.9040961791372522,
           rho=0.6506770477876749,
           loglikelihood=-7281.321453218112)

Use :class:`frds.algorithms.GARCHModel_DCC` to estimate a bivariate :doc:`/algorithms/garch-dcc` model.
The results are as good as those obtained in Stata/R, if not better (based on loglikelihood).

>>> from frds.algorithms import GARCHModel_DCC
>>> model_dcc = GARCHModel_DCC(toyota, nissan)
>>> res = model_dcc.fit()
>>> from pprint import pprint
>>> pprint(res)
Parameters(mu1=0.039598837827953585,
           omega1=0.027895534722110118,
           alpha1=0.06942955278530698,
           beta1=0.9216715294923623,
           mu2=0.019315543596552513,
           omega2=0.05701047522984261,
           alpha2=0.0904653253307871,
           beta2=0.8983752570013462,
           a=0.04305972552559641,
           b=0.894147940765443,
           loglikelihood=-7256.572183143142)

