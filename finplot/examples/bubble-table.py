#!/usr/bin/env python3
'''Plots quotes as bubbles and as a table on the second axis. The table
   is composed of labels (text) and a heatmap (background color).'''

import dateutil.parser
import finplot as fplt
import numpy as np
import pandas as pd
import requests


interval_mins = 1
start_t = '2022-01-05T19:15Z'
count = 35
downsample = 3
start_ts = int(dateutil.parser.parse(start_t).timestamp()) + 1*60


def download_resample():
    end_ts = start_ts + (count*downsample-2)*60
    price_url = f'https://www.bitmex.com/api/udf/history?symbol=XBTUSD&resolution={interval_mins}&from={start_ts}&to={end_ts}'
    quote_url = f'https://www.bitmex.com/api/v1/quote/bucketed?symbol=XBT&binSize={interval_mins}m&startTime={start_t}&count={count*downsample}'

    prices = pd.DataFrame(requests.get(price_url).json())
    quotes = pd.DataFrame(requests.get(quote_url).json())
    prices['timestamp'] = pd.to_datetime(prices.t, unit='s')
    quotes['timestamp'] = pd.to_datetime(quotes.timestamp)
    prices.set_index('timestamp', inplace=True)
    quotes.set_index('timestamp', inplace=True)
    prices, quotes = resample(prices, quotes)
    return prices, quotes


def resample(prices, quotes):
    quotes.bidPrice = (quotes.bidPrice*quotes.bidSize).rolling(downsample).sum() / quotes.bidSize.rolling(downsample).sum()
    quotes.bidSize = quotes.bidSize.rolling(downsample).sum()
    quotes.askPrice = (quotes.askPrice*quotes.askSize).rolling(downsample).sum() / quotes.askSize.rolling(downsample).sum()
    quotes.askSize = quotes.askSize.rolling(downsample).sum()
    q = quotes.iloc[downsample-1::downsample]
    q.index = quotes.index[::downsample]

    p = prices.rename(columns={'o':'Open', 'c':'Close', 'h':'High', 'l':'Low', 'v':'Volume'})
    p.Open = p.Open.shift(downsample-1)
    p.High = p.High.rolling(downsample).max()
    p.Low = p.Low.rolling(downsample).min()
    p.Volume = p.Volume.rolling(downsample).sum()
    p = p.iloc[downsample-1::downsample]
    p.index = q.index

    return p,q


def plot_bubble_pass(price, price_col, size_col, min_val, max_val, scale, color, ax):
    price = price.copy()
    price.loc[(price[size_col]<min_val)|(price[size_col]>max_val), price_col] = np.nan
    fplt.plot(price[price_col], style='o', width=scale, color=color, ax=ax)


def plot_quote_bubbles(quotes, ax):
    quotes['bidSize2'] = np.sqrt(quotes.bidSize) # linearize by circle area
    quotes['askSize2'] = np.sqrt(quotes.askSize)
    size2 = quotes.bidSize2.append(quotes.askSize2)
    rng = np.linspace(size2.min(), size2.max(), 5)
    rng = list(zip(rng[:-1], rng[1:]))
    for a,b in reversed(rng):
        scale = (a+b) / rng[-1][1] + 0.2
        plot_bubble_pass(quotes, 'bidPrice', 'bidSize2', a, b, scale=scale, color='#0f0', ax=ax)
        plot_bubble_pass(quotes, 'askPrice', 'askSize2', a, b, scale=scale, color='#f00', ax=ax)


def plot_quote_table(quotes, ax):
    '''Plot quote table (in millions). We're using lables on top of a heatmap to create sort of a table.'''
    ax.set_visible(yaxis=False) # Y axis is useless on our table
    def skip_y_crosshair_info(x, y, xt, yt): # we don't want any Y crosshair info on the table
        return xt, ''
    fplt.add_crosshair_info(skip_y_crosshair_info, ax=ax)
    fplt.set_y_range(0, 2, ax) # 0-1 for bid row, 1-2 for ask row

    # add two columns for table cell colors
    quotes[1] = -quotes['askSize'] * 0.5 / quotes['askSize'].max() + 0.5
    quotes[0] = +quotes['bidSize'] * 0.5 / quotes['bidSize'].max() + 0.5

    ts = [int(t.timestamp()) for t in quotes.index]
    colmap = fplt.ColorMap([0.0, 0.5, 1.0], [[200, 80, 60], [200, 190, 100], [40, 170, 30]]) # traffic light colors
    fplt.heatmap(quotes[[1, 0]], colmap=colmap, colcurve=lambda x: x, ax=ax) # linear color mapping
    fplt.labels(ts, [1.5]*count, ['%.1f'%(v/1e6) for v in quotes['askSize']], ax=ax2, anchor=(0.5, 0.5))
    fplt.labels(ts, [0.5]*count, ['%.1f'%(v/1e6) for v in quotes['bidSize']], ax=ax2, anchor=(0.5, 0.5))


prices, quotes = download_resample()

fplt.max_zoom_points = 5
fplt.right_margin_candles = 0
ax,ax2 = fplt.create_plot(f'BitMEX {downsample}m quote bubble plot + quote table', rows=2, maximize=False)
fplt.windows[0].ci.layout.setRowStretchFactor(0, 10) # make primary plot large, and implicitly table small
candles = fplt.candlestick_ochl(prices[['Open','Close','High','Low']], ax=ax)
candles.colors.update(dict(bear_body='#fa8')) # bright red, to make bubbles visible
fplt.volume_ocv(prices[['Open','Close','Volume']], ax=ax.overlay())
plot_quote_bubbles(quotes, ax=ax)
plot_quote_table(quotes, ax=ax2)
fplt.show()
