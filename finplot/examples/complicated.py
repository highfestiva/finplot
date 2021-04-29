#!/usr/bin/env python3
'''A lengthy example that shows some more complex uses of finplot:
    - control panel in PyQt
    - varying indicators, intervals and layout
    - toggle dark mode
    - price line
    - real-time updates via websocket

   This example includes dipping in to the internals of finplot and
   the underlying lib pyqtgraph, which is not part of the API per se,
   and may thus change in the future. If so happens, this example
   will be updated to reflect such changes.

   Included is also some third-party libraries to make the example
   more realistic.
   '''


import finplot as fplt
from functools import lru_cache
import json
from math import nan
import pandas as pd
from PyQt5.QtWidgets import QComboBox, QCheckBox, QWidget
from pyqtgraph import QtGui
import pyqtgraph as pg
import requests
from time import time as now, sleep
from threading import Thread
import websocket


class BinanceFutureWebsocket:
    def __init__(self):
        self.url = 'wss://fstream.binance.com/stream'
        self.symbol = None
        self.interval = None
        self.ws = None
        self.df = None

    def reconnect(self, symbol, interval, df):
        '''Connect and subscribe, if not already done so.'''
        self.df = df
        if symbol.lower() == self.symbol and self.interval == interval:
            return
        self.symbol = symbol.lower()
        self.interval = interval
        self.thread_connect = Thread(target=self._thread_connect)
        self.thread_connect.daemon = True
        self.thread_connect.start()

    def close(self, reset_symbol=True):
        if reset_symbol:
            self.symbol = None
        if self.ws:
            self.ws.close()
        self.ws = None

    def _thread_connect(self):
        self.close(reset_symbol=False)
        print('websocket connecting to %s...' % self.url)
        self.ws = websocket.WebSocketApp(self.url, on_message=self.on_message, on_error=self.on_error)
        self.thread_io = Thread(target=self.ws.run_forever)
        self.thread_io.daemon = True
        self.thread_io.start()
        for _ in range(100):
            if self.ws.sock and self.ws.sock.connected:
                break
            sleep(0.1)
        else:
            self.close()
            raise websocket.WebSocketTimeoutException('websocket connection failed')
        self.subscribe(self.symbol, self.interval)
        print('websocket connected')

    def subscribe(self, symbol, interval):
        try:
            data = '{"method":"SUBSCRIBE","params":["%s@kline_%s"],"id":1}' % (symbol, interval)
            self.ws.send(data)
        except Exception as e:
            print('websocket subscribe error:', type(e), e)
            raise e

    def on_message(self, msg):
        df = self.df
        if df is None:
            return
        msg = json.loads(msg)
        if 'stream' not in msg:
            return
        stream = msg['stream']
        if '@kline_' in stream:
            k = msg['data']['k']
            t = k['t']
            t0 = int(df.index[-2].timestamp()) * 1000
            t1 = int(df.index[-1].timestamp()) * 1000
            t2 = t1 + (t1-t0)
            if t < t2:
                # update last candle
                i = df.index[-1]
                df.loc[i, 'Close']  = float(k['c'])
                df.loc[i, 'High']   = max(df.loc[i, 'High'], float(k['h']))
                df.loc[i, 'Low']    = min(df.loc[i, 'Low'],  float(k['l']))
                df.loc[i, 'Volume'] = float(k['v'])
            else:
                # create a new candle
                data = [t] + [float(k[i]) for i in ['o','c','h','l','v']]
                candle = pd.DataFrame([data], columns='Time Open Close High Low Volume'.split()).astype({'Time':'datetime64[ms]'})
                candle.set_index('Time', inplace=True)
                self.df = df.append(candle)

    def on_error(self, error):
        print('websocket error: %s' % error)


def do_load_price_history(symbol, interval):
    url = 'https://www.binance.com/fapi/v1/klines?symbol=%s&interval=%s&limit=%s' % (symbol, interval, 1000)
    print('loading binance future %s %s' % (symbol, interval))
    d = requests.get(url).json()
    df = pd.DataFrame(d, columns='Time Open High Low Close Volume a b c d e f'.split())
    df = df.astype({'Time':'datetime64[ms]', 'Open':float, 'High':float, 'Low':float, 'Close':float, 'Volume':float})
    return df.set_index('Time')


@lru_cache(maxsize=5)
def cache_load_price_history(symbol, interval):
    '''Stupid caching, but works sometimes.'''
    return do_load_price_history(symbol, interval)


def load_price_history(symbol, interval):
    '''Use memoized, and if too old simply load the data.'''
    df = cache_load_price_history(symbol, interval)
    # check if cache's newest candle is current
    t0 = df.index[-2].timestamp()
    t1 = df.index[-1].timestamp()
    t2 = t1 + (t1 - t0)
    if now() >= t2:
        df = do_load_price_history(symbol, interval)
    return df


def calc_parabolic_sar(df, af=0.2, steps=10):
    up = True
    sars = [nan] * len(df)
    sar = ep_lo = df.Low.iloc[0]
    ep = ep_hi = df.High.iloc[0]
    aaf = af
    aaf_step = aaf / steps
    af = 0
    for i,(hi,lo) in enumerate(zip(df.High, df.Low)):
        # parabolic sar formula:
        sar = sar + af * (ep - sar)
        # handle new extreme points
        if hi > ep_hi:
            ep_hi = hi
            if up:
                ep = ep_hi
                af = min(aaf, af+aaf_step)
        elif lo < ep_lo:
            ep_lo = lo
            if not up:
                ep = ep_lo
                af = min(aaf, af+aaf_step)
        # handle switch
        if up:
            if lo < sar:
                up = not up
                sar = ep_hi
                ep = ep_lo = lo
                af = 0
        else:
            if hi > sar:
                up = not up
                sar = ep_lo
                ep = ep_hi = hi
                af = 0
        sars[i] = sar
    df['sar'] = sars
    return df['sar']


def calc_rsi(price, n=14, ax=None):
    diff = price.diff().values
    gains = diff
    losses = -diff
    gains[~(gains>0)] = 0.0
    losses[~(losses>0)] = 1e-10 # we don't want divide by zero/NaN
    m = (n-1) / n
    ni = 1 / n
    g = gains[n] = gains[:n].mean()
    l = losses[n] = losses[:n].mean()
    gains[:n] = losses[:n] = nan
    for i,v in enumerate(gains[n:],n):
        g = gains[i] = ni*v + m*g
    for i,v in enumerate(losses[n:],n):
        l = losses[i] = ni*v + m*l
    rs = gains / losses
    rsi = 100 - (100/(1+rs))
    return rsi


def calc_stochastic_oscillator(df, n=14, m=3, smooth=3):
    lo = df.Low.rolling(n).min()
    hi = df.High.rolling(n).max()
    k = 100 * (df.Close-lo) / (hi-lo)
    d = k.rolling(m).mean()
    return k, d


def calc_plot_data(df, indicators):
    '''Returns data for all plots and for the price line.'''
    price = df['Open Close High Low'.split()]
    volume = df['Open Close Volume'.split()]
    ma50 = ma200 = vema24 = sar = rsi = stoch = stoch_s = None
    if 'few' in indicators or 'moar' in indicators:
        ma50  = price.Close.rolling(50).mean()
        ma200 = price.Close.rolling(200).mean()
        vema24 = volume.Volume.ewm(span=24).mean()
    if 'moar' in indicators:
        sar = calc_parabolic_sar(df)
        rsi = calc_rsi(df.Close)
        stoch,stoch_s = calc_stochastic_oscillator(df)
    plot_data = dict(price=price, volume=volume, ma50=ma50, ma200=ma200, vema24=vema24, sar=sar, rsi=rsi, \
                     stoch=stoch, stoch_s=stoch_s)
    # for price line
    last_close = price.iloc[-1].Close
    last_col = fplt.candle_bull_color if last_close > price.iloc[-2].Close else fplt.candle_bear_color
    price_data = dict(last_close=last_close, last_col=last_col)
    return plot_data, price_data


def realtime_update_plot():
    '''Called at regular intervals by a timer.'''
    if ws.df is None:
        return

    # calculate the new plot data
    indicators = ctrl_panel.indicators.currentText().lower()
    data,price_data = calc_plot_data(ws.df, indicators)

    # first update all data, then graphics (for zoom rigidity)
    for k in data:
        if data[k] is not None:
            plots[k].update_data(data[k], gfx=False)
    for k in data:
        if data[k] is not None:
            plots[k].update_gfx()

    # place and color price line
    ax.price_line.setPos(price_data['last_close'])
    ax.price_line.pen.setColor(pg.mkColor(price_data['last_col']))


def change_asset(*args, **kwargs):
    '''Resets and recalculates everything, and plots for the first time.'''
    # save window zoom position before resetting
    fplt._savewindata(fplt.windows[0])

    symbol = ctrl_panel.symbol.currentText()
    interval = ctrl_panel.interval.currentText()
    ws.df = None
    df = load_price_history(symbol, interval=interval)
    ws.reconnect(symbol, interval, df)

    # remove any previous plots
    ax.reset()
    axo.reset()
    ax_rsi.reset()

    # calculate plot data
    indicators = ctrl_panel.indicators.currentText().lower()
    data,price_data = calc_plot_data(df, indicators)

    # some space for legend
    ctrl_panel.move(100 if 'clean' in indicators else 200, 0)

    # plot data
    global plots
    plots = {}
    plots['price'] = fplt.candlestick_ochl(data['price'], ax=ax)
    plots['volume'] = fplt.volume_ocv(data['volume'], ax=axo)
    if data['ma50'] is not None:
        plots['ma50'] = fplt.plot(data['ma50'], legend='MA-50', ax=ax)
        plots['ma200'] = fplt.plot(data['ma200'], legend='MA-200', ax=ax)
        plots['vema24'] = fplt.plot(data['vema24'], color=4, legend='V-EMA-24', ax=axo)
    if data['rsi'] is not None:
        ax.set_visible(xaxis=False)
        ax_rsi.show()
        fplt.set_y_range(0, 100, ax=ax_rsi)
        fplt.add_band(30, 70, color='#6335', ax=ax_rsi)
        plots['sar'] = fplt.plot(data['sar'], color='#55a', style='+', width=0.6, legend='SAR', ax=ax)
        plots['rsi'] = fplt.plot(data['rsi'], legend='RSI', ax=ax_rsi)
        plots['stoch'] = fplt.plot(data['stoch'], color='#880', legend='Stoch', ax=ax_rsi)
        plots['stoch_s'] = fplt.plot(data['stoch_s'], color='#650', ax=ax_rsi)
    else:
        ax.set_visible(xaxis=True)
        ax_rsi.hide()

    # price line
    ax.price_line = pg.InfiniteLine(angle=0, movable=False, pen=fplt._makepen(fplt.candle_bull_body_color, style='.'))
    ax.price_line.setPos(price_data['last_close'])
    ax.price_line.pen.setColor(pg.mkColor(price_data['last_col']))
    ax.addItem(ax.price_line, ignoreBounds=True)

    # restores saved zoom position, if in range
    fplt.refresh()


def dark_mode_toggle(dark):
    '''Digs into the internals of finplot and pyqtgraph to change the colors of existing
       plots, axes, backgronds, etc.'''
    # first set the colors we'll be using
    if dark:
        fplt.foreground = '#777'
        fplt.background = '#090c0e'
        fplt.candle_bull_color = fplt.candle_bull_body_color = '#0b0'
        fplt.candle_bear_color = '#a23'
        volume_transparency = '6'
    else:
        fplt.foreground = '#444'
        fplt.background = fplt.candle_bull_body_color = '#fff'
        fplt.candle_bull_color = '#380'
        fplt.candle_bear_color = '#c50'
        volume_transparency = 'c'
    fplt.volume_bull_color = fplt.volume_bull_body_color = fplt.candle_bull_color + volume_transparency
    fplt.volume_bear_color = fplt.candle_bear_color + volume_transparency
    fplt.cross_hair_color = fplt.foreground+'8'
    fplt.draw_line_color = '#888'
    fplt.draw_done_color = '#555'

    pg.setConfigOptions(foreground=fplt.foreground, background=fplt.background)
    # control panel color
    if ctrl_panel is not None:
        p = ctrl_panel.palette()
        p.setColor(ctrl_panel.darkmode.foregroundRole(), pg.mkColor(fplt.foreground))
        ctrl_panel.darkmode.setPalette(p)

    # window background
    for win in fplt.windows:
        win.setBackground(fplt.background)

    # axis, crosshair, candlesticks, volumes
    axs = [ax for win in fplt.windows for ax in win.axs]
    vbs = set([ax.vb for ax in axs])
    axs += fplt.overlay_axs
    axis_pen = fplt._makepen(color=fplt.foreground)
    for ax in axs:
        ax.axes['left']['item'].setPen(axis_pen)
        ax.axes['left']['item'].setTextPen(axis_pen)
        ax.axes['bottom']['item'].setPen(axis_pen)
        ax.axes['bottom']['item'].setTextPen(axis_pen)
        if ax.crosshair is not None:
            ax.crosshair.vline.pen.setColor(pg.mkColor(fplt.foreground))
            ax.crosshair.hline.pen.setColor(pg.mkColor(fplt.foreground))
            ax.crosshair.xtext.setColor(fplt.foreground)
            ax.crosshair.ytext.setColor(fplt.foreground)
        for item in ax.items:
            if isinstance(item, fplt.FinPlotItem):
                isvolume = ax in fplt.overlay_axs
                if not isvolume:
                    item.colors.update(
                        dict(bull_shadow      = fplt.candle_bull_color,
                             bull_frame       = fplt.candle_bull_color,
                             bull_body        = fplt.candle_bull_body_color,
                             bear_shadow      = fplt.candle_bear_color,
                             bear_frame       = fplt.candle_bear_color,
                             bear_body        = fplt.candle_bear_color))
                else:
                    item.colors.update(
                        dict(bull_frame       = fplt.volume_bull_color,
                             bull_body        = fplt.volume_bull_body_color,
                             bear_frame       = fplt.volume_bear_color,
                             bear_body        = fplt.volume_bear_color))
                item.repaint()


def create_ctrl_panel(win):
    panel = QWidget(win)
    panel.move(100, 0)
    win.scene().addWidget(panel)
    layout = QtGui.QGridLayout(panel)

    panel.symbol = QComboBox(panel)
    [panel.symbol.addItem(i+'USDT') for i in 'BTC ETH XRP DOGE BNB SOL ADA LTC LINK DOT TRX BCH'.split()]
    panel.symbol.setCurrentIndex(1)
    layout.addWidget(panel.symbol, 0, 0)
    panel.symbol.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(1, 30)

    panel.interval = QComboBox(panel)
    [panel.interval.addItem(i) for i in '1d 4h 1h 30m 15m 5m 1m'.split()]
    panel.interval.setCurrentIndex(6)
    layout.addWidget(panel.interval, 0, 2)
    panel.interval.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(3, 30)

    panel.indicators = QComboBox(panel)
    [panel.indicators.addItem(i) for i in 'Clean:Few indicators:Moar indicators'.split(':')]
    panel.indicators.setCurrentIndex(1)
    layout.addWidget(panel.indicators, 0, 4)
    panel.indicators.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(5, 30)

    panel.darkmode = QCheckBox(panel)
    panel.darkmode.setText('Haxxor mode')
    panel.darkmode.setCheckState(2)
    panel.darkmode.toggled.connect(dark_mode_toggle)
    layout.addWidget(panel.darkmode, 0, 6)

    return panel


plots = {}
fplt.y_pad = 0.07 # pad some extra (for control panel)
fplt.max_zoom_points = 7
fplt.autoviewrestore()
ax,ax_rsi = fplt.create_plot('Complicated Binance Futures Example', rows=2, init_zoom_periods=300)
axo = ax.overlay()

# use websocket for real-time
ws = BinanceFutureWebsocket()

# hide rsi chart to begin with; show x-axis of top plot
ax_rsi.hide()
ax_rsi.vb.setBackgroundColor(None) # don't use odd background color
ax.set_visible(xaxis=True)

ctrl_panel = create_ctrl_panel(ax.vb.win)
dark_mode_toggle(True)
change_asset()
fplt.timer_callback(realtime_update_plot, 1) # update every second
fplt.show()
