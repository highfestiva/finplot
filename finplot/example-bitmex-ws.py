#!/usr/bin/env python3
'''Slightly more advanced example which requires "pip install bitmex-ws" to run. The data is pushed from the BitMEX server
   to our chart, which we update once per second.'''

from bitmex_websocket import BitMEXWebsocket
import dateutil.parser
import finplot as fplt
import pandas as pd
import requests
from time import time


baseurl = 'https://www.bitmex.com/api'
plots = []


def price_history(load_periods=500, interval_mins=1):
    symbol = 'XBTUSD'
    end_t = int(time()) + 120*interval_mins
    end_t -= end_t % (60*interval_mins)
    start_t = end_t - load_periods*60*interval_mins
    url = baseurl + '/udf/history?symbol=%s&resolution=%s&from=%s&to=%s' % (symbol, interval_mins, start_t, end_t)
    d = requests.get(url).json()
    # drop volume and status
    del d['v']
    del d['s']
    return d


def calc_bollinger_bands(df):
    r = df['c'].rolling(20)
    df['bbh'] = r.mean() + 2*r.std()
    df['bbl'] = r.mean() - 2*r.std()


def update_plot(df):
    calc_bollinger_bands(df)
    datasrc0 = fplt.PandasDataSource(df['t o c h l'.split()])
    datasrc1 = fplt.PandasDataSource(df['t bbh'.split()])
    datasrc2 = fplt.PandasDataSource(df['t bbl'.split()])
    if not plots:
        candlestick_plot = fplt.candlestick_ochl(datasrc0, ax=ax)
        plots.append(candlestick_plot)
        plots.append(fplt.plot_datasrc(datasrc1, color='#4e4ef1', ax=ax))
        plots.append(fplt.plot_datasrc(datasrc2, color='#4e4ef1', ax=ax))
        # redraw using bitmex colors
        candlestick_plot.bull_color = '#388d53'
        candlestick_plot.bull_frame_color = '#205536'
        candlestick_plot.bull_body_color = '#52b370'
        candlestick_plot.bear_color = '#d56161'
        candlestick_plot.bear_frame_color = '#5c1a10'
        candlestick_plot.bear_body_color = '#e8704f'
        candlestick_plot.repaint()
    else:
        plots[0].update_datasrc(datasrc0)
        plots[1].update_datasrc(datasrc1)
        plots[2].update_datasrc(datasrc2)


def update_data(interval_mins=1):
    global df, ws
    for trade in ws.recent_trades():
        t = int(dateutil.parser.parse(trade['timestamp']).timestamp())
        t -= t % (60*interval_mins)
        c = trade['price']
        if t < df['t'].iloc[-1]:
            # ignore already-recorded trades
            continue
        elif t > df['t'].iloc[-1]:
            # add new candle
            o = df['c'].iloc[-1]
            h = c if c>o else o
            l = o if o<c else c
            df1 = pd.DataFrame(dict(t=[t], o=[o], c=[c], h=[l], l=[l]))
            df = pd.concat([df, df1], ignore_index=True, sort=False)
        else:
            # update last candle
            i = df.index.max()
            df.loc[i,'c'] = c
            if c > df.loc[i,'h']:
                df.loc[i,'h'] = c
            if c < df.loc[i,'l']:
                df.loc[i,'l'] = c
    update_plot(df)


if __name__ == '__main__':
    df = pd.DataFrame(price_history())
    ws = BitMEXWebsocket(endpoint=baseurl+'/v1', symbol='XBTUSD')
    ax = fplt.create_plot('Realtime Bitcoin/Dollar 1m (BitMEX websocket)', init_zoom_periods=100, maximize=False)
    update_plot(df)
    fplt.timer_callback(update_data, 1.0) # update every second
    fplt.show()
