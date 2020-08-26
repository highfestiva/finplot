#!/usr/bin/env python3

import finplot as fplt
import yfinance as yf

df = yf.download('BNO', '2014-01-01')
fplt.create_plot('US Brent Oil Renko', maximize=False)
fplt.renko(df.Close)
fplt.show()
