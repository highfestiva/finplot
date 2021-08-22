#!/usr/bin/env python3

import finplot as fplt
from functools import lru_cache
from PyQt5.QtWidgets import QApplication, QGridLayout, QMainWindow, QGraphicsView, QComboBox, QLabel
from pyqtgraph.dockarea import DockArea, Dock
from threading import Thread
import yfinance as yf

app = QApplication([])
win = QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1600,800)
win.setWindowTitle("Docking charts example for finplot")

# Set width/height of QSplitter
win.setStyleSheet("QSplitter { width : 20px; height : 20px; }")

# Create docks
dock_0 = Dock("dock_0", size = (1000, 100), closable = True)
dock_1 = Dock("dock_1", size = (1000, 100), closable = True)
dock_2 = Dock("dock_2", size = (1000, 100), closable = True)
area.addDock(dock_0)
area.addDock(dock_1)
area.addDock(dock_2)

# Create example charts
combo = QComboBox()
combo.setEditable(True)
[combo.addItem(i) for i in "AMRK FB GFN REVG TSLA TWTR WMT CT=F GC=F ^FTSE ^N225 EURUSD=X ETH-USD".split()]
dock_0.addWidget(combo, 0, 0, 1, 1)
info = QLabel()
dock_0.addWidget(info, 0, 1, 1, 1)

# Chart for dock_0
ax0,ax1,ax2 = fplt.create_plot_widget(master=area, rows=3, init_zoom_periods=100)
area.axs = [ax0, ax1, ax2]
dock_0.addWidget(ax0.ax_widget, 1, 0, 1, 2)
dock_1.addWidget(ax1.ax_widget, 1, 0, 1, 2)
dock_2.addWidget(ax2.ax_widget, 1, 0, 1, 2)

# Link x-axis
ax1.setXLink(ax0)
ax2.setXLink(ax0)
win.axs = [ax0]

@lru_cache(maxsize = 15)
def download(symbol):
    return yf.download(symbol, "2019-01-01")

@lru_cache(maxsize = 100)
def get_name(symbol):
    return yf.Ticker(symbol).info ["shortName"]

plots = []
def update(txt):
    df = download(txt)
    if len(df) < 20: # symbol does not exist
        return
    info.setText("Loading symbol name...")
    price = df ["Open Close High Low".split()]
    ma20 = df.Close.rolling(20).mean()
    ma50 = df.Close.rolling(50).mean()
    volume = df ["Open Close Volume".split()]
    ax0.reset() # remove previous plots
    ax1.reset() # remove previous plots
    ax2.reset() # remove previous plots
    fplt.candlestick_ochl(price, ax = ax0)
    fplt.plot(ma20, legend = "MA-20", ax = ax1)
    fplt.plot(ma50, legend = "MA-50", ax = ax1)
    fplt.volume_ocv(volume, ax = ax2)
    fplt.refresh() # refresh autoscaling when all plots complete
    Thread(target=lambda: info.setText(get_name(txt))).start() # slow, so use thread

combo.currentTextChanged.connect(update)
update(combo.currentText())

fplt.show(qt_exec = False) # prepares plots when they're all setup
win.show()
app.exec_()
