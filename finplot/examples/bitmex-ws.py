#!/usr/bin/env python3
'''Slightly more advanced example which requires "pip install bitmex-ws" to run. The data is pushed from the BitMEX server
   to our chart, which we update in a couple of Hertz.'''

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
orderbook = None


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
    global orderbook
    calc_bollinger_bands(df)
    candlesticks = df['t o c h l'.split()]
    bollband_hi = df['t bbh'.split()]
    bollband_lo = df['t bbl'.split()]
    if not plots: # 1st time
        candlestick_plot = fplt.candlestick_ochl(candlesticks)
        plots.append(candlestick_plot)
        plots.append(fplt.plot(bollband_hi, color='#4e4ef1'))
        plots.append(fplt.plot(bollband_lo, color='#4e4ef1'))
        fplt.fill_between(plots[1], plots[2], color='#9999fa')
        # generate dummy orderbook plot, which we update next time
        x = len(candlesticks)+0.5
        y = candlesticks.c.iloc[-1]
        orderbook = [[x,[(y,1)]]]
        orderbook_colorfunc = fplt.horizvol_colorfilter([(0,'bull'),(10,'bear')])
        orderbook_plot = fplt.horiz_time_volume(orderbook, candle_width=1, draw_body=10, colorfunc=orderbook_colorfunc)
        plots.append(orderbook_plot)
        # use bitmex colors
        candlestick_plot.colors.update(dict(
                bull_shadow = '#388d53',
                bull_frame  = '#205536',
                bull_body   = '#52b370',
                bear_shadow = '#d56161',
                bear_frame  = '#5c1a10',
                bear_body   = '#e8704f'))
        orderbook_plot.colors.update(dict(
                bull_frame  = '#52b370',
                bull_body   = '#bae1c6',
                bear_frame  = '#e8704f',
                bear_body   = '#f6c6b9'))
    else: # update
        plots[0].update_data(candlesticks)
        plots[1].update_data(bollband_hi)
        plots[2].update_data(bollband_lo)
        plots[3].update_data(orderbook)


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
        o = c = h = l = df['c'].iloc[-1]
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


def update_orderbook_data(orderbook10):
    global orderbook
    ask = pd.DataFrame(orderbook10['asks'], columns=['price','volume'])
    bid = pd.DataFrame(orderbook10['bids'], columns=['price','volume'])
    bid['price'] -= 0.5 # bid below price
    ask['volume'] = -ask['volume'].cumsum() # negative volume means pointing left
    bid['volume'] = -bid['volume'].cumsum()
    orderbook = [[len(df)+0.5, pd.concat([bid.iloc[::-1], ask])]]


if __name__ == '__main__':
    df = pd.DataFrame(price_history())

    # fix bug in BitMEX websocket lib
    def bugfix(self, *args, **kwargs):
        from pyee import EventEmitter
        EventEmitter.__init__(self)
        orig_init(self, *args, **kwargs)
    orig_init = Instrument.__init__
    Instrument.__init__ = bugfix

    ws = Instrument(channels=[InstrumentChannels.trade, InstrumentChannels.orderBook10])
    @ws.on('action')
    def action(message):
        if 'orderBook' in message['table']:
            for orderbook10 in message['data']:
                update_orderbook_data(orderbook10)
        else:
            for trade in message['data']:
                update_candlestick_data(trade)
    thread = Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()
    fplt.create_plot('Realtime Bitcoin/Dollar 1m (BitMEX websocket)', init_zoom_periods=75, maximize=False)
    update_plot()
    fplt.timer_callback(update_plot, 0.5) # update in 2 Hz
    fplt.show()
