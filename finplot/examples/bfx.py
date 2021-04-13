#!/usr/bin/env python3

import math
import pandas as pd
import finplot as fplt
import requests
import time


def cumcnt_indices(v):
    v[~v] = math.nan
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

    # calculate indicator
    tdup,tddn = td_sequential(df['close'])
    df['tdup'] = [('%i'%i if 0<i<10 else '') for i in tdup]
    df['tddn'] = [('%i'%i if 0<i<10 else '') for i in tddn]

    # pick columns for our three data sources: candlesticks and TD sequencial labels for up/down
    candlesticks = df['time open close high low'.split()]
    volumes = df['time open close volume'.split()]
    td_up_labels = df['time high tdup'.split()]
    td_dn_labels = df['time low tddn'.split()]
    if not plots:
        # first time we create the plots
        global ax
        plots.append(fplt.candlestick_ochl(candlesticks))
        plots.append(fplt.volume_ocv(volumes, ax=ax.overlay()))
        plots.append(fplt.labels(td_up_labels, color='#009900'))
        plots.append(fplt.labels(td_dn_labels, color='#990000', anchor=(0.5,0)))
    else:
        # every time after we just update the data sources on each plot
        plots[0].update_data(candlesticks)
        plots[1].update_data(volumes)
        plots[2].update_data(td_up_labels)
        plots[3].update_data(td_dn_labels)


plots = []
ax = fplt.create_plot('Realtime Bitcoin/Dollar 1m TD Sequential (BitFinex REST)', init_zoom_periods=100, maximize=False)
update()
fplt.timer_callback(update, 5.0) # update (using synchronous rest call) every N seconds

fplt.show()
