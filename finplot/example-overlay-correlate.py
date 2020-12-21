#!/usr/bin/env python3

from datetime import date, timedelta
import finplot as fplt
import pandas as pd
import scipy.optimize
import yfinance as yf

now = date.today()
start_day = now - timedelta(days=55)
df = yf.download('GOOG', start_day.isoformat(), now.isoformat(), interval='90m')
dfms = yf.download('MSFT', start_day.isoformat(), now.isoformat(), interval='90m')

# resample to daily candles, i.e. five 90-minute candles per business day
dfd = df.Open.resample('D').first().to_frame()
dfd['Close'] = df.Close.resample('D').last()
dfd['High'] = df.High.resample('D').max()
dfd['Low'] = df.Low.resample('D').min()

ax,ax2 = fplt.create_plot('Alphabet Inc.', rows=2, maximize=False)
ax2.disable_x_index() # second plot is not timebased

# plot down-sampled daily candles first
daily_plot = fplt.candlestick_ochl(dfd.dropna(), candle_width=5)
daily_plot.colors.update(dict(bull_body='#bfb', bull_shadow='#ada', bear_body='#fbc', bear_shadow='#dab'))
daily_plot.x_offset = 3.1 # resample() gets us start of day, offset +1.1 (gap+off center wick)

# plot high resolution on top
fplt.candlestick_ochl(df[['Open','Close','High','Low']])

# scatter plot correlation between Google and Microsoft stock
df['ret_alphabet']  = df.Close.pct_change()
df['ret_microsoft'] = dfms.Close.pct_change()
dfc = df.dropna().reset_index(drop=True)[['ret_alphabet', 'ret_microsoft']]
fplt.plot(dfc, style='o', color=1, ax=ax2)

# draw least-square line
errfun = lambda arr: [y-arr[0]*x+arr[1] for x,y in zip(dfc.ret_alphabet, dfc.ret_microsoft)]
line = scipy.optimize.least_squares(errfun, [0.01, 0.01]).x
linex = [dfc.ret_alphabet.min(), dfc.ret_alphabet.max()]
liney = [linex[0]*line[0]+line[1], linex[1]*line[0]+line[1]]
fplt.add_line((linex[0],liney[0]), (linex[1],liney[1]), color='#993', ax=ax2)
fplt.add_text((linex[1],liney[1]), 'k=%.2f'%line[0], color='#993', ax=ax2)
fplt.add_legend('Alphabet vs. Microsft 90m correlation', ax=ax2)

fplt.show()
