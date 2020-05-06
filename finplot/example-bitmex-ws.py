#!/usr/bin/env python3
'''Slightly more advanced example which requires "pip install bitmex-ws" to run. The data is pushed from the BitMEX server
   to our chart, which we update once per second.'''

from bitmex_websocket import Instrument
from bitmex_websocket.constants import InstrumentChannels
import dateutil.parser
import finplot as fplt
import pandas as pd
import requests
from threading import Thread
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


def update_plot():
    calc_bollinger_bands(df)
    candlesticks = df['t o c h l'.split()]
    bollband_hi = df['t bbh'.split()]
    bollband_lo = df['t bbl'.split()]
    if not plots:
        candlestick_plot = fplt.candlestick_ochl(candlesticks)
        plots.append(candlestick_plot)
        plots.append(fplt.plot(bollband_hi, color='#4e4ef1'))
        plots.append(fplt.plot(bollband_lo, color='#4e4ef1'))
        fplt.fill_between(plots[1], plots[2], color='#9999fa')
        # redraw using bitmex colors
        candlestick_plot.colors.update(dict(
                bull_shadow = '#388d53',
                bull_frame  = '#205536',
                bull_body   = '#52b370',
                bear_shadow = '#d56161',
                bear_frame  = '#5c1a10',
                bear_body   = '#e8704f'))
        candlestick_plot.repaint()
    else:
        plots[0].update_data(candlesticks)
        plots[1].update_data(bollband_hi)
        plots[2].update_data(bollband_lo)


def update_candlestick_data(trade, interval_mins=1):
    global df
    t = int(dateutil.parser.parse(trade['timestamp']).timestamp())
    t -= t % (60*interval_mins)
    c = trade['price']
    if t < df['t'].iloc[-1]:
        # ignore already-recorded trades
        return
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


if __name__ == '__main__':
    df = pd.DataFrame(price_history())
    ws = Instrument(channels=[InstrumentChannels.trade])
    @ws.on('action')
    def action(message):
        if not 'data' in message:
            return
        for trade in message['data']:
            update_candlestick_data(trade)
    thread = Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()
    fplt.create_plot('Realtime Bitcoin/Dollar 1m (BitMEX websocket)', init_zoom_periods=100, maximize=False)
    update_plot()
    fplt.timer_callback(update_plot, 1.0) # update every second
    fplt.show()
