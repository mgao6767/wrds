=============
Stock Returns
=============

stock_us
========

>>> from frds.datasets import StockReturns
>>> returns = StockReturns.stocks_us

:class:`frds.datasets.StockReturns.stocks_us` provides daily stock returns of a few U.S. 
stocks, including Google, Goldman Sachs, JPMorgan, and the S&P500 index, from 2010 to 2022.

>>> returns.head()
               GOOGL        GS       JPM     ^GSPC
Date                                              
2010-01-05 -0.004404  0.017680  0.019370  0.003116
2010-01-06 -0.025209 -0.010673  0.005494  0.000546
2010-01-07 -0.023280  0.019569  0.019809  0.004001
2010-01-08  0.013331 -0.018912 -0.002456  0.002882
2010-01-11 -0.001512 -0.015776 -0.003357  0.001747
>>> len(returns)
3271

Below is a visualization of the returns and indexed prices.

.. image:: /images/stocks_us.png
