#!/usr/bin/env python3

from collections import defaultdict
import dateutil.parser
import finplot as fplt
import pandas as pd
import requests


baseurl = 'https://www.bitmex.com/api'


def local2timestamp(s):
    return int(dateutil.parser.parse(s).timestamp())

def download_price_history(symbol='XBTUSD', start_time='2019-04-01', end_time='2019-07-07', interval_mins=60):
    start_time = local2timestamp(start_time)
    end_time = local2timestamp(end_time)
    data = defaultdict(list)
    for start_t in range(start_time, end_time, 10000*interval_mins):
        end_t = start_t + 10000*interval_mins
        if end_t > end_time:
            end_t = end_time
        url = baseurl + '/udf/history?symbol=%s&resolution=%s&from=%s&to=%s' % (symbol, interval_mins, start_t, end_t)
        print(url)
        d = requests.get(url).json()
        del d['s'] # ignore status=ok
        for col in d:
            data[col] += d[col]
    return pd.DataFrame(data)


symbol = 'XBTUSD'
df = download_price_history(symbol=symbol)
ax,axv = fplt.create_plot('BitMEX %s price history' % symbol, rows=2)
candle_datasrc = fplt.PandasDataSource(df['t o c h l'.split()])
fplt.candlestick_ochl(candle_datasrc, ax=ax)
volume_datasrc = fplt.PandasDataSource(df['t o c v'.split()])
fplt.volume_ocv(volume_datasrc, ax=axv)
fplt.show()
