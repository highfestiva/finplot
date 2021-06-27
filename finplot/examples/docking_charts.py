#!/usr/bin/env python3

import finplot as fplt
from functools import lru_cache
from PyQt5.QtWidgets import QGraphicsView, QComboBox, QLabel
from PyQt5.QtGui import QApplication, QGridLayout
from PyQt5 import QtGui
from pyqtgraph.dockarea import DockArea, Dock
from threading import Thread
import yfinance as yf

app = QApplication ([])
win = QtGui.QMainWindow ()
area = DockArea ()
win.setCentralWidget (area)
win.resize (1600,800)
win.setWindowTitle ("Docking charts example for finplot")

# Set width/height of QSplitter
win.setStyleSheet ("QSplitter { width : 20px; height : 20px; }")

# Create docks
dock_0 = Dock ("dock_0", size = (1000, 100), closable = True)
dock_1 = Dock ("dock_1", size = (1000, 100), closable = True)
dock_2 = Dock ("dock_2", size = (1000, 100), closable = True)
area.addDock (dock_0)
area.addDock (dock_1)
area.addDock (dock_2)

# Create example charts
combo = QComboBox ()
combo.setEditable (True)
[combo.addItem (i) for i in "AMRK FB GFN REVG TSLA TWTR WMT CT=F GC=F ^FTSE ^N225 EURUSD=X ETH-USD".split ()]
dock_0.addWidget (combo, 0, 0, 1, 1)
info = QLabel ()
dock_0.addWidget (info, 0, 1, 1, 1)

# Chart for dock_0
ax_0 = fplt.create_plot (init_zoom_periods = 100)
ax_0.set_visible(xgrid=True, ygrid=True)
dock_0.addWidget (ax_0.vb.win, 1, 0, 1, 2)

# Chart for dock_1
ax_1 = fplt.create_plot (init_zoom_periods = 100)
dock_1.addWidget (ax_1.vb.win, 1, 0, 1, 2)

# Chart for dock_2
ax_2 = fplt.create_plot (init_zoom_periods = 100)
dock_2.addWidget (ax_2.vb.win, 1, 0, 1, 2)

# Link x-axis
ax_1.setXLink (ax_0)
ax_2.setXLink (ax_0)
win.axs = [ax_0]

@lru_cache (maxsize = 15)
def download (symbol):
    return yf.download (symbol, "2019-01-01")

@lru_cache (maxsize = 100)
def get_name (symbol):
    return yf.Ticker (symbol).info ["shortName"]

plots = []
def update (txt):
    df = download (txt)
    if len (df) < 20: # symbol does not exist
        return
    info.setText ("Loading symbol name...")
    price = df ["Open Close High Low".split ()]
    ma20 = df.Close.rolling (20).mean ()
    ma50 = df.Close.rolling (50).mean ()
    volume = df ["Open Close Volume".split ()]
    ax_0.reset () # remove previous plots
    ax_1.reset() # remove previous plots
    ax_2.reset() # remove previous plots
    fplt.candlestick_ochl (price, ax = ax_0)
    fplt.plot (ma20, legend = "MA-20", ax = ax_1)
    fplt.plot (ma50, legend = "MA-50", ax = ax_1)
    fplt.volume_ocv (volume, ax = ax_2)
    fplt.refresh () # refresh autoscaling when all plots complete
    Thread (target=lambda: info.setText (get_name (txt))).start () # slow, so use thread

combo.currentTextChanged.connect (update)
update (combo.currentText ())

fplt.show (qt_exec = False) # prepares plots when they"re all setup
win.show()
app.exec_()
