#!/usr/bin/env python3
'''A lengthy example that shows some more complex uses of finplot:
    - control panel in PyQt
    - varying indicators, intervals, layout & colors

   This example includes digging in to the internals of finplot and
   the underlying lib pyqtgraph, which is not part of the API per se,
   and may thus change in the future. If so happens, this example
   will be updated to reflect such changes.'''


import finplot as fplt
from functools import lru_cache
from math import nan
import pandas as pd
from PyQt5.QtWidgets import QComboBox, QCheckBox, QWidget
from pyqtgraph import QtGui
import pyqtgraph as pg
import requests
from time import time as now


ctrl_panel = None


def do_load_price_history(symbol, interval):
    url = 'https://www.binance.com/fapi/v1/klines?symbol=%s&interval=%s&limit=%s' % (symbol, interval, 1000)
    print('loading binance future: %s' % url)
    d = requests.get(url).json()
    df = pd.DataFrame(d, columns='Time Open High Low Close Volume a b c d e f'.split())
    df = df.astype({'Time':'datetime64[ms]', 'Open':float, 'High':float, 'Low':float, 'Close':float, 'Volume':float})
    return df.set_index('Time')


@lru_cache(maxsize=5)
def cache_load(symbol, interval):
    return now(), do_load_price_history(symbol, interval)


def load_price_history(symbol, interval):
    t, df = cache_load(symbol, interval)
    # check if cache is older than N seconds
    if now()-t > 30:
        df = do_load_price_history(symbol, interval)
    return df


def plot_parabolic_sar(df, af=0.2, steps=10, ax=None):
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
    p = df.sar.plot(ax=ax, color='#55a', style='+', width=0.6, legend='SAR')


def plot_rsi(price, n=14, ax=None):
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
    fplt.plot(rsi, ax=ax, legend='RSI')
    fplt.set_y_range(0, 100, ax=ax)
    fplt.add_band(30, 70, color='#6335', ax=ax)


def plot_stochastic_oscillator(df, n=14, m=3, smooth=3, ax=None):
    lo = df.Low.rolling(n).min()
    hi = df.High.rolling(n).max()
    k = 100 * (df.Close-lo) / (hi-lo)
    d = k.rolling(m).mean()
    fplt.plot(k.rolling(smooth).mean(), color='#880', legend='Stoch', ax=ax)
    fplt.plot(d.rolling(smooth).mean(), color='#650', ax=ax)


def change_asset(*args, **kwargs):
    # save window zoom position before resetting
    fplt._savewindata(fplt.windows[0])

    symbol = ctrl_panel.symbol.currentText()
    interval = ctrl_panel.interval.currentText()
    df = load_price_history(symbol, interval=interval)
    price = df['Open Close High Low'.split()]
    volume = df['Open Close Volume'.split()]

    # remove any previous plots
    ax.reset()
    axo.reset()
    ax_rsi.reset()

    fplt.candlestick_ochl(price, ax=ax)
    fplt.volume_ocv(volume, ax=axo)

    indicators = ctrl_panel.indicators.currentText().lower()
    clean = 'clean' in indicators
    ctrl_panel.move(100 if clean else 200, 0)
    if not clean:
        ma50  = price.Close.rolling(50).mean()
        ma200 = price.Close.rolling(200).mean()
        fplt.plot(ma50, legend='MA-50', ax=ax)
        fplt.plot(ma200, legend='MA-200', ax=ax)
        vma20 = volume.Volume.rolling(20).mean()
        fplt.plot(vma20, color=4, ax=axo, legend='VMA-20')
    if 'many' in indicators:
        ax.set_visible(xaxis=False)
        ax_rsi.show()
        plot_parabolic_sar(df, ax=ax)
        plot_rsi(df.Close, ax=ax_rsi)
        plot_stochastic_oscillator(df, ax=ax_rsi)
    else:
        ax.set_visible(xaxis=True)
        ax_rsi.hide()

    # restores saved zoom position, if in range
    fplt.refresh()


def dark_mode_toggle(dark):
    '''Digs into the internals of finplot and pyqtgraph to change the colors of existing
       plots, axes, backgronds, etc.'''
    if dark:
        fplt.foreground = '#777'
        fplt.background = '#090c0e'
        fplt.candle_bull_color = fplt.candle_bull_body_color = '#0b0'
        fplt.candle_bear_color = '#a23'
        volume_transparency = '6'
        fplt.draw_line_color = '#fff'
        fplt.draw_done_color = '#aaa'
    else:
        fplt.foreground = '#444'
        fplt.background = fplt.candle_bull_body_color = '#fff'
        fplt.candle_bull_color = '#380'
        fplt.candle_bear_color = '#c50'
        volume_transparency = 'c'
        fplt.draw_line_color = '#000'
        fplt.draw_done_color = '#555'
    fplt.volume_bull_color = fplt.volume_bull_body_color = fplt.candle_bull_color + volume_transparency
    fplt.volume_bear_color = fplt.candle_bear_color + volume_transparency
    fplt.cross_hair_color = fplt.foreground+'8'

    pg.setConfigOptions(foreground=fplt.foreground, background=fplt.background)
    axs = [ax for win in fplt.windows for ax in win.axs]
    vbs = set([ax.vb for ax in axs])
    axs += fplt.overlay_axs
    axis_pen = fplt._makepen(color=fplt.foreground)
    if ctrl_panel is not None:
        p = ctrl_panel.palette()
        p.setColor(ctrl_panel.darkmode.foregroundRole(), pg.mkColor(fplt.foreground))
        p.setColor(ctrl_panel.backgroundRole(), pg.mkColor(fplt.background))
        ctrl_panel.darkmode.setPalette(p)
        ctrl_panel.setPalette(p)
    for win in fplt.windows:
        win.setBackground(fplt.background)
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
    layout.addWidget(panel.symbol, 0, 0)
    panel.symbol.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(1, 30)

    panel.interval = QComboBox(panel)
    [panel.interval.addItem(i) for i in '1d 4h 1h 30m 15m 5m 1m'.split()]
    layout.addWidget(panel.interval, 0, 2)
    panel.interval.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(3, 30)

    panel.indicators = QComboBox(panel)
    [panel.indicators.addItem(i) for i in 'Clean:Few indicators:Many indicators'.split(':')]
    layout.addWidget(panel.indicators, 0, 4)
    panel.indicators.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(5, 30)

    panel.darkmode = QCheckBox(panel)
    panel.darkmode.setText('Dark mode')
    panel.darkmode.setCheckState(2)
    panel.darkmode.toggled.connect(dark_mode_toggle)
    layout.addWidget(panel.darkmode, 0, 6)

    return panel


fplt.y_pad = 0.07 # pad some more (for control panel)
fplt.max_zoom_points = 7
fplt.autoviewrestore()
ax,ax_rsi = fplt.create_plot('Big', rows=2, init_zoom_periods=1000)
axo = ax.overlay()

# hide rsi chart to begin with; show x-axis of top plot
ax_rsi.hide()
ax_rsi.vb.setBackgroundColor(None)
ax.set_visible(xaxis=True)

ctrl_panel = create_ctrl_panel(ax.vb.win)
dark_mode_toggle(True)
change_asset()
fplt.show()
