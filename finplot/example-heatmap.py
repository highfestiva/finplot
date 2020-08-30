#!/usr/bin/env python3

import finplot as fplt
import pandas as pd
import requests


# download, extract times and candles
url = 'https://www.tensorcharts.com/tensor/bitmex/XBTUSD/heatmapCandles/15min'
data = requests.get(url).json()
times = pd.to_datetime([e['T'] for e in data])
candles = [(e['open'],e['close'],e['high'],e['low']) for e in data]
df_candles = pd.DataFrame(index=times, data=candles)

# extract volume heatmap as a PRICE x VOLUME matrix
orderbooks = [(e['heatmapOrderBook'] or []) for e in data]
prices = sorted(set(prc for ob in orderbooks for prc in ob[::2]))
vol_matrix = [[0]*len(prices) for _ in range(len(times))]
for i,orderbook in enumerate(orderbooks):
    for price,volume in zip(orderbook[::2],orderbook[1::2]):
        j = prices.index(price)
        vol_matrix[i][j] = volume
df_volume_heatmap = pd.DataFrame(index=times, columns=prices, data=vol_matrix)

# plot
fplt.create_plot('BitMEX BTC 15m orderbook heatmap')
fplt.candlestick_ochl(df_candles)
fplt.heatmap(df_volume_heatmap, filter_limit=0.2, whiteout=0.3)
fplt.show()
