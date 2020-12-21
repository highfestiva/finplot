#!/usr/bin/env python3

import finplot as fplt
import numpy as np
import pandas as pd


FPS = 30
anim_counter = 0
spots = None
labels_plot = None


def gen_dumb_price():
    # start with four random columns
    v = np.random.normal(size=(1000,4))
    df = pd.DataFrame(v, columns='low open close high'.split())
    # smooth out
    df = df.rolling(10).mean()
    # add a bit of push
    ma = df['low'].rolling(20).mean().diff()
    for col in df.columns:
        df[col] *= ma*100
    # arrange so low is lowest each period, high highest, open and close in between
    df.values.sort(axis=1)
    # add price variation over time and some amplitude
    df = (df.T + np.sin(df.index/87) * 3 + np.cos(df.index/201) * 5).T + 20
    # some green, some red candles
    flip = df['open'].shift(-1) <= df['open']
    df.loc[flip,'open'],df.loc[flip,'close'] = df['close'].copy(),df['open'].copy()
    # price action => volume
    df['volume'] = df['high'] - df['low']
    # epoch time
    df.index = np.linspace(1608332400-60*1000, 1608332400, 1000)
    return df['open close high low volume'.split()].dropna()


def gen_spots(ax, df):
    global spots
    spot_ser = df['low'] - 0.1
    spot_ser[(spot_ser.reset_index(drop=True).index - anim_counter) % 20 != 0] = np.nan
    if spots is None:
        spots = spot_ser.plot(kind='scatter', color=2, width=2, ax=ax, zoomscale=False)
    else:
        spots.update_data(spot_ser)


def gen_labels(ax, df):
    global labels_plot
    y_ser = df['volume'] - 0.1
    y_ser[(y_ser.reset_index(drop=True).index + anim_counter) % 50 != 0] = np.nan
    dft = y_ser.to_frame()
    dft.columns = ['y']
    dft['text'] = dft['y'].apply(lambda v: str(round(v, 1)) if v>0 else '')
    if labels_plot is None:
        labels_plot = dft.plot(kind='labels', ax=ax)
    else:
        labels_plot.update_data(dft)


def move_view(ax, df):
    global anim_counter
    x = -np.cos(anim_counter/100)*(len(df)/2-50) + len(df)/2
    w = np.sin(anim_counter/100)**4*50 + 50
    fplt.set_x_pos(df.index[int(x-w)], df.index[int(x+w)], ax=ax)
    anim_counter += 1


def animate(ax, ax2, df):
    gen_spots(ax, df)
    gen_labels(ax2, df)
    move_view(ax, df)


df = gen_dumb_price()
ax,ax2 = fplt.create_plot('Things move', rows=2, init_zoom_periods=100, maximize=False)
df.plot(kind='candle', ax=ax)
df[['open','close','volume']].plot(kind='volume', ax=ax2)
fplt.timer_callback(lambda: animate(ax, ax2, df), 1/FPS)
fplt.show()
