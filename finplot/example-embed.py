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
[combo.addItem(i) for i in 'AMRK FB GFN REVG TWTR WMT CT=F GC=F ^FTSE ^N225 EURUSD=X ETH-USD'.split()]
layout.addWidget(combo, 0, 0, 1, 1)
info = QLabel()
layout.addWidget(info, 0, 1, 1, 1)

ax = fplt.create_plot_widget(win, init_zoom_periods=100)
win.axs = [ax] # finplot requres this property
layout.addWidget(ax.ax_widget, 1, 0, 1, 2)


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
    if not plots:
        plots.append(fplt.candlestick_ochl(price))
        plots.append(fplt.plot(ma20, legend='MA-20'))
        plots.append(fplt.plot(ma50, legend='MA-50'))
        plots.append(fplt.volume_ocv(volume, ax=ax.overlay()))
    else:
        plots[0].update_data(price)
        plots[1].update_data(ma20)
        plots[2].update_data(ma50)
        plots[3].update_data(volume)
    Thread(target=lambda: info.setText(get_name(txt))).start() # slow, so use thread

combo.currentTextChanged.connect(update)
update(combo.currentText())


fplt.show(qt_exec=False) # prepares plots when they're all setup
win.show()
app.exec_()
