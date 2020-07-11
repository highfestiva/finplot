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
df['Date'] = pd.to_datetime(df['Date']).astype('int64') // 1_000_000 # use finplot's internal representation, which is ms

# plot candles
ax,ax2 = fplt.create_plot('S&P 500 MACD', rows=2)
fplt.candlestick_ochl(df[['Date','Open','Close','High','Low']], ax=ax)
hover_label = fplt.add_legend('', ax=ax)
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

def update_legend_text(x, y):
    row = df.loc[df.Date==x]
    # format html with the candle and set legend
    fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open<row.Close).all() else 'a00')
    rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
    hover_label.setText(rawtxt % (symbol, interval.upper(), row.Open, row.Close, row.High, row.Low))

def update_crosshair_text(ax, x, y, xtext, ytext):
    ytext = '%s (Close%+.2f)' % (ytext, (y - df.iloc[x].Close))
    return xtext, ytext

fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
fplt.add_crosshair_info(update_crosshair_text, ax=ax)

fplt.show()
