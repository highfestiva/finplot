#!/usr/bin/env python3

from collections import defaultdict
import dateutil.parser
import finplot as fplt
import pandas as pd
import pytz
import requests


utc2timestamp = lambda s: int(dateutil.parser.parse(s).replace(tzinfo=pytz.utc).timestamp() * 1000)


def download_price_history(symbol='BTCUSDT', start_time='2020-06-22', end_time='2020-08-19', interval_mins=1):
    interval_ms = 1000*60*interval_mins
    interval_str = '%sm'%interval_mins if interval_mins<60 else '%sh'%(interval_mins//60)
    start_time = utc2timestamp(start_time)
    end_time = utc2timestamp(end_time)
    data = []
    for start_t in range(start_time, end_time, 1000*interval_ms):
        end_t = start_t + 1000*interval_ms
        if end_t >= end_time:
            end_t = end_time - interval_ms
        url = 'https://www.binance.com/fapi/v1/klines?interval=%s&limit=%s&symbol=%s&startTime=%s&endTime=%s' % (interval_str, 1000, symbol, start_t, end_t)
        print(url)
        d = requests.get(url).json()
        data += d
    df = pd.DataFrame(data, columns='time open high low close volume a b c d e f'.split())
    return df.astype({'time':'datetime64[ms]', 'open':float, 'high':float, 'low':float, 'close':float, 'volume':float})


def calc_volume_profile(df, period, bins):
    '''Calculate a poor man's volume distribution/profile by "pinpointing" each kline volume to a certain
       price and placing them, into N buckets. (IRL volume would be something like "trade-bins" per candle.)
       The output format is a matrix, where each [period] time is a row index, and even columns contain
       start (low) price and odd columns contain volume (for that price and time interval). See
       finplot.horiz_time_volume() for more info.'''
    data = []
    df['hlc3'] = (df.high + df.low + df.close) / 3 # assume this is volume center per each 1m candle
    _,all_bins = pd.cut(df.hlc3, bins, right=False, retbins=True)
    for _,g in df.groupby(pd.Grouper(key='time', freq=period)):
        t = g.time.iloc[0]
        volbins = pd.cut(g.hlc3, all_bins, right=False)
        price2vol = defaultdict(float)
        for iv,vol in zip(volbins, g.volume):
            price2vol[iv.left] += vol
        data.append([t, sorted(price2vol.items())])
    return data


def calc_vwap(period):
    vwap = pd.Series ([], dtype = 'float64')
    df['hlc3v'] = df['hlc3'] * df.volume
    for _,g in df.groupby(pd.Grouper(key='time', freq=period)):
        i0,i1 = g.index[0],g.index[-1]
        vwap = vwap.append(g.hlc3v.loc[i0:i1].cumsum() / df.volume.loc[i0:i1].cumsum())
    return vwap


# download and calculate indicators
df = download_price_history(interval_mins=30) # reduce to [15, 5, 1] minutes to increase accuracy
time_volume_profile = calc_volume_profile(df, period='W', bins=100) # try fewer/more horizontal bars (graphical resolution only)
vwap = calc_vwap(period='W') # try period='D'

# plot
fplt.create_plot('Binance BTC futures weekly volume profile')
fplt.plot(df.time, df.close, legend='Price')
fplt.plot(df.time, vwap, style='--', legend='VWAP')
fplt.horiz_time_volume(time_volume_profile, draw_va=0.7, draw_poc=1.0)
fplt.show()
