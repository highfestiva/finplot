#!/usr/bin/env python3

import finplot as fplt
import pandas as pd
import requests
from io import StringIO
from time import time


# load data and convert date
end_t = int(time()) 
start_t = end_t - 12*30*24*60*60 # twelve months
symbol = 'SPY'
interval = '1d'
url = 'https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=%s&events=history' % (symbol, start_t, end_t, interval)
r = requests.get(url)
df = pd.read_csv(StringIO(r.text))
df['Date'] = pd.to_datetime(df['Date'])

# plot candles
ax,ax2 = fplt.create_plot('S&P 500 MACD', rows=2)
fplt.candlestick_ochl(df[['Date','Open','Close','High','Low']], ax=ax)
axo = ax.overlay()
fplt.volume_ocv(df[['Date','Open','Close','Volume']], ax=axo)
fplt.plot(df.Volume.ewm(span=24).mean(), ax=axo, color=1)

# plot macd
macd = df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean()
signal = macd.ewm(span=9).mean()
df['macd_diff'] = macd - signal
fplt.volume_ocv(df[['Date','Open','Close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
fplt.plot(macd, ax=ax2, legend='MACD')
fplt.plot(signal, ax=ax2, legend='Signal')

fplt.show()
