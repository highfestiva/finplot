#!/usr/bin/env python3

import finplot as fplt
from functools import lru_cache
from PyQt5.QtWidgets import QGraphicsView, QComboBox, QLabel
from PyQt5.QtGui import QApplication, QGridLayout
from threading import Thread
import yfinance as yf


app = QApplication([])
win = QGraphicsView()
win.setWindowTitle('TradingView wannabe')
layout = QGridLayout()
win.setLayout(layout)
win.resize(600, 500)

combo = QComboBox()
combo.setEditable(True)
[combo.addItem(i) for i in 'AMRK FB GFN REVG TSLA TWTR WMT CT=F GC=F ^FTSE ^N225 EURUSD=X ETH-USD'.split()]
layout.addWidget(combo, 0, 0, 1, 1)
info = QLabel()
layout.addWidget(info, 0, 1, 1, 1)

ax = fplt.create_plot(init_zoom_periods=100)
win.axs = [ax] # finplot requres this property
axo = ax.overlay()
layout.addWidget(ax.vb.win, 1, 0, 1, 2)


@lru_cache(maxsize=15)
def download(symbol):
    return yf.download(symbol, '2019-01-01')

@lru_cache(maxsize=100)
def get_name(symbol):
    return yf.Ticker(symbol).info['shortName']

plots = []
def update(txt):
    df = download(txt)
    if len(df) < 20: # symbol does not exist
        return
    info.setText('Loading symbol name...')
    price = df['Open Close High Low'.split()]
    ma20 = df.Close.rolling(20).mean()
    ma50 = df.Close.rolling(50).mean()
    volume = df['Open Close Volume'.split()]
    ax.reset() # remove previous plots
    axo.reset() # remove previous plots
    fplt.candlestick_ochl(price)
    fplt.plot(ma20, legend='MA-20')
    fplt.plot(ma50, legend='MA-50')
    fplt.volume_ocv(volume, ax=axo)
    fplt.refresh() # refresh autoscaling when all plots complete
    Thread(target=lambda: info.setText(get_name(txt))).start() # slow, so use thread

combo.currentTextChanged.connect(update)
update(combo.currentText())


fplt.show(qt_exec=False) # prepares plots when they're all setup
win.show()
app.exec_()
