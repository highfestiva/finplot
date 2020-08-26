#!/usr/bin/env python3

import finplot as fplt
import yfinance as yf

df = yf.download('BTC-USD', '2014-01-01')
import numpy as np
df['c'] = df.Close#np.log(df.Close)
fplt.renko(df.c, bins=100)
fplt.show()
