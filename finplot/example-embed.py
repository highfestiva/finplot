#!/usr/bin/env python3

import finplot as fplt
from functools import lru_cache
from PyQt5.QtWidgets import QGraphicsView, QComboBox
from PyQt5.QtGui import QApplication, QGridLayout
import yfinance as yf


app = QApplication([])
win = QGraphicsView()
win.setWindowTitle('TradingView wannabe')
layout = QGridLayout()
win.setLayout(layout)
win.resize(600, 500)

combo = QComboBox()
combo.setEditable(True)
combo.addItem('SPY')
combo.addItem('BNO')
combo.addItem('CT=F')
combo.addItem('AAPL')
combo.addItem('WMT')
combo.addItem('GOOG')
combo.addItem('BTC-USD')
combo.addItem('ETH-USD')
layout.addWidget(combo, 0, 0, 1, 1)

ax = fplt.create_plot_widget(win)
win.axs = [ax] # finplot requres this property
layout.addWidget(ax.ax_widget, 1, 0, 1, 1)


@lru_cache(maxsize=15)
def download(symbol):
    return yf.download(symbol, '2019-01-01')

plots = []
def update(txt):
    df = download(txt)
    if len(df) < 20: # symbol does not exist
        return
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
combo.currentTextChanged.connect(update)
update(combo.currentText())


fplt.show(qt_exec=False) # prepares plots when they're all setup
win.show()
app.exec_()
