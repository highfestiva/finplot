#!/usr/bin/env python3

import pandas as pd
import finplot as fplt
import requests
import time


def cumcnt_indices(v):
    v[~v] = pd.np.nan
    cumsum = v.cumsum().fillna(method='pad')
    reset = -cumsum[v.isnull()].diff().fillna(cumsum)
    r = v.where(v.notnull(), reset).cumsum().fillna(0.0)
    return r.astype(int)


def td_sequential(close):
    close4 = close.shift(4)
    td = cumcnt_indices(close > close4)
    ts = cumcnt_indices(close < close4)
    return td, ts


def update():
    # load data
    limit = 500
    start = int(time.time()*1000) - (500-2)*60*1000
    url = 'https://api.bitfinex.com/v2/candles/trade:1m:tBTCUSD/hist?limit=%i&sort=1&start=%i' % (limit, start)
    table = requests.get(url).json()
    df = pd.DataFrame(table, columns='time open close high low volume'.split())
    df['time'] //= 1000 # convert ms to seconds

    # calculate indicator
    tdup,tddn = td_sequential(df['close'])
    df['tdup'] = [('%i'%i if 0<i<10 else '') for i in tdup]
    df['tddn'] = [('%i'%i if 0<i<10 else '') for i in tddn]

    # pick columns for our three data sources: candlesticks and the gree
    datasrc0 = fplt.PandasDataSource(df['time open close high low'.split()])
    datasrc1 = fplt.PandasDataSource(df['time high tdup'.split()])
    datasrc2 = fplt.PandasDataSource(df['time low tddn'.split()])
    if not plots:
        # first time we create the plots
        plots.append(fplt.candlestick_ochl(datasrc0, ax=ax))
        plots.append(fplt.labels_datasrc(datasrc1, color='#009900', ax=ax))
        plots.append(fplt.labels_datasrc(datasrc2, color='#990000', ax=ax, anchor=(0.5,0)))
    else:
        # every time after we just update the data sources on each plot
        plots[0].update_datasrc(datasrc0)
        plots[1].update_datasrc(datasrc1)
        plots[2].update_datasrc(datasrc2)


plots = []
ax = fplt.create_plot('Realtime Bitcoin/Dollar 1m (BitFinex)', init_zoom_periods=100, maximize=False)
update()
fplt.timer_callback(update, 20.0) # update every N seconds

fplt.show()
