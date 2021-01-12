#!/usr/bin/env python3

from collections import defaultdict
import dateutil.parser
import finplot as fplt
import numpy as np
import pandas as pd
import requests


baseurl = 'https://www.bitmex.com/api'


def local2timestamp(s):
    return int(dateutil.parser.parse(s).timestamp())


def download_price_history(symbol='XBTUSD', start_time='2020-01-10', end_time='2020-05-07', interval_mins=60):
    start_time = local2timestamp(start_time)
    end_time = local2timestamp(end_time)
    data = defaultdict(list)
    for start_t in range(start_time, end_time, 10000*60*interval_mins):
        end_t = start_t + 10000*60*interval_mins
        if end_t > end_time:
            end_t = end_time
        url = '%s/udf/history?symbol=%s&resolution=%s&from=%s&to=%s' % (baseurl, symbol, interval_mins, start_t, end_t)
        print(url)
        d = requests.get(url).json()
        del d['s'] # ignore status=ok
        for col in d:
            data[col] += d[col]
    df = pd.DataFrame(data)
    df = df.rename(columns={'t':'time', 'o':'open', 'c':'close', 'h':'high', 'l':'low', 'v':'volume'})
    return df.set_index('time')


def plot_accumulation_distribution(df, ax):
    ad = (2*df.close-df.high-df.low) * df.volume / (df.high - df.low)
    ad.cumsum().ffill().plot(ax=ax, legend='Accum/Dist', color='#f00000')


def plot_bollinger_bands(df, ax):
    mean = df.close.rolling(20).mean()
    stddev = df.close.rolling(20).std()
    df['boll_hi'] = mean + 2.5*stddev
    df['boll_lo'] = mean - 2.5*stddev
    p0 = df.boll_hi.plot(ax=ax, color='#808080', legend='BB')
    p1 = df.boll_lo.plot(ax=ax, color='#808080')
    fplt.fill_between(p0, p1, color='#bbb')


def plot_ema(df, ax):
    df.close.ewm(span=9).mean().plot(ax=ax, legend='EMA')


def plot_heikin_ashi(df, ax):
    df['h_close'] = (df.open+df.close+df.high+df.low) / 4
    ho = (df.open.iloc[0] + df.close.iloc[0]) / 2
    for i,hc in zip(df.index, df['h_close']):
        df.loc[i, 'h_open'] = ho
        ho = (ho + hc) / 2
    print(df['h_open'])
    df['h_high'] = df[['high','h_open','h_close']].max(axis=1)
    df['h_low'] = df[['low','h_open','h_close']].min(axis=1)
    df[['h_open','h_close','h_high','h_low']].plot(ax=ax, kind='candle')


def plot_heikin_ashi_volume(df, ax):
    df[['h_open','h_close','volume']].plot(ax=ax, kind='volume')


def plot_on_balance_volume(df, ax):
    obv = df.volume.copy()
    obv[df.close < df.close.shift()] = -obv
    obv[df.close==df.close.shift()] = 0
    obv.cumsum().plot(ax=ax, legend='OBV', color='#008800')


def plot_rsi(df, ax):
    diff = df.close.diff().values
    gains = diff
    losses = -diff
    with np.errstate(invalid='ignore'):
        gains[(gains<0)|np.isnan(gains)] = 0.0
        losses[(losses<=0)|np.isnan(losses)] = 1e-10 # we don't want divide by zero/NaN
    n = 14
    m = (n-1) / n
    ni = 1 / n
    g = gains[n] = np.nanmean(gains[:n])
    l = losses[n] = np.nanmean(losses[:n])
    gains[:n] = losses[:n] = np.nan
    for i,v in enumerate(gains[n:],n):
        g = gains[i] = ni*v + m*g
    for i,v in enumerate(losses[n:],n):
        l = losses[i] = ni*v + m*l
    rs = gains / losses
    df['rsi'] = 100 - (100/(1+rs))
    df.rsi.plot(ax=ax, legend='RSI')
    fplt.set_y_range(0, 100, ax=ax)
    fplt.add_band(30, 70, ax=ax)


def plot_vma(df, ax):
    df.volume.rolling(20).mean().plot(ax=ax, color='#c0c030')


symbol = 'XBTUSD'
df = download_price_history(symbol=symbol)

ax,axv,ax2,ax3,ax4 = fplt.create_plot('BitMEX %s heikin-ashi price history' % symbol, rows=5)
ax.set_visible(xgrid=True, ygrid=True)

# price chart
plot_heikin_ashi(df, ax)
plot_bollinger_bands(df, ax)
plot_ema(df, ax)

# volume chart
plot_heikin_ashi_volume(df, axv)
plot_vma(df, ax=axv)

# some more charts
plot_accumulation_distribution(df, ax2)
plot_on_balance_volume(df, ax3)
plot_rsi(df, ax4)

# restore view (X-position and zoom) when we run this example again
fplt.autoviewrestore()

fplt.show()
