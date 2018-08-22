# -*- coding: utf-8 -*-
'''
Financial data plotter with better defaults, api, behavior and performance than
mpl_finance and plotly.

Lines up your time-series with a shared X-axis; ideal for volume, RSI, etc.

Zoom does something similar to what you'd normally expect for financial data,
where the Y-axis is auto-scaled to highest high and lowest low in the active
region.
'''

from datetime import datetime
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


legend_border_color = '#000000dd'
legend_fill_color   = '#00000088'
legend_text_color   = '#dddddd66'
prev_ax = None

_epoch2local = lambda t: datetime.fromtimestamp(t).isoformat().replace('T',' ')



class EpochAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [_epoch2local(value) for value in values]



class PandasDataSource:
    '''Candle sticks: create with five columns: time, open, close, hi, lo - in that order.
       Volume bars: create with three columns: time, open, close, volume - in that order.
       For all other types, time needs to be first, usually followed by one or more Y-columns.'''
    def __init__(self, df):
        self.df = df.copy()
        timecol = self.df.columns[0]
        self.df[timecol] = _pdtime2epoch(df[timecol])
        self.skip_scale_colcnt = 1 # skip at least time, for candle sticks volume we also skip open and close

    @property
    def period(self):
        timecol = self.df.columns[0]
        return self.df[timecol].iloc[1] - self.df[timecol].iloc[0]

    @property
    def x(self):
        timecol = self.df.columns[0]
        return self.df[timecol]

    @property
    def y(self):
        ycol = self.df.columns[1]
        return self.df[ycol]

    def addcols(self, df):
        newcols = df[df.columns[1:]] # skip timecol
        self.df = pd.concat([self.df, newcols], axis=1)

    def get_time(self, offset_from_end=0, period=0):
        '''Return timestamp of offset *from end*.'''
        if offset_from_end >= len(self.df):
            offset_from_end = len(self.df)-1
        timecol = self.df.columns[0]
        t = self.df[timecol].iloc[-1-offset_from_end]
        if period:
            t += period * self.period
        return t

    def hilo(self, x0, x1):
        '''Return five values in time range: t0, t1, highest, lowest, number of rows.'''
        df = self.df
        timecol = df.columns[0]
        df = df.loc[((df[timecol]>=x0)&(df[timecol]<=x1)), :]
        t0 = df[timecol].iloc[0]
        t1 = df[timecol].iloc[-1]
        valcols = df.columns[self.skip_scale_colcnt:]
        hi = df[valcols].max().max()
        lo = df[valcols].min().min()
        return t0,t1,hi,lo,len(df)

    def bear_rows(self):
        opencol = self.df.columns[1]
        closecol = self.df.columns[2]
        rows = self.df.loc[(self.df.loc[:,opencol]>self.df.loc[:,closecol])] # open higher than close = goes down
        return zip(*[rows[c] for c in rows.columns])

    def bull_rows(self):
        opencol = self.df.columns[1]
        closecol = self.df.columns[2]
        rows = self.df.loc[(self.df.loc[:,opencol]<=self.df.loc[:,closecol])] # open lower than close = goes up
        return zip(*[rows[c] for c in rows.columns])



class FinLegendItem(pg.LegendItem):
    def __init__(self, border_color, fill_color, **kwargs):
        super().__init__(**kwargs)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(2, 2, 10, 2)
        self.border_color = border_color
        self.fill_color = fill_color

    def paint(self, p, *args):
        p.setPen(pg.mkPen(self.border_color))
        p.setBrush(pg.mkBrush(self.fill_color))
        p.drawRect(self.boundingRect())



class FinViewBox(pg.ViewBox):
    def __init__(self, win, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win = win
        self.set_datasrc(None)
        self.setMouseEnabled(x=True, y=False)
        self.auto_scale_y = True

    def set_datasrc(self, datasrc, init_steps=300):
        self.datasrc = datasrc
        if not self.datasrc:
            return
        x0 = datasrc.get_time(offset_from_end=init_steps, period=-0.5)
        x1 = datasrc.get_time(offset_from_end=0, period=+0.5)
        t0,t1,hi,lo,cnt = self.datasrc.hilo(x0, x1)
        self.setRange(QtCore.QRectF(pg.Point(t0, lo), pg.Point(t1, hi)), padding=0)

    def wheelEvent(self, ev, axis=None):
        scale_fact = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        vr = self.targetRect()
        center = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(ev.pos()))
        if (center.x()-vr.left())/vr.width() < 0.2: # zoom to far left => all the way left
            center = pg.Point(vr.left(), center.y())
        elif (center.x()-vr.left())/vr.width() > 0.8: # zoom to far right => all the way right
            center = pg.Point(vr.right(), center.y())
        self.scaleRect(vr, scale_fact, center)
        ev.accept()

    def scale(self, scale_fact):
        self.scaleRect(vr, scale_fact, None)

    def linkedViewChanged(self, view, axis):
        if view is not None:
            tr = self.targetRect()
            vr = view.viewRect()
            if not self.auto_scale_y:
                vr = QtCore.QRectF(pg.Point(vr.left(), tr.top()), pg.Point(vr.right(), tr.bottom()))
            self.scaleRect(vr, 1.0)

    def scaleRect(self, vr, scale_fact, center=None):
        if not self.datasrc:
            return
        x_ = vr.left()
        if center is None:
            center = vr.center()
        x0 = center.x() + (vr.left()-center.x()) * scale_fact
        x1 = center.x() + (vr.right()-center.x()) * scale_fact
        t0,t1,hi,lo,cnt = self.datasrc.hilo(x0, x1)
        if cnt < 20:
            return
        x0 = t0 - self.datasrc.period*0.5
        x1 = t1 + self.datasrc.period*0.5
        if not self.auto_scale_y:
            lo = vr.top()
            hi = vr.bottom()
        self.setRange(QtCore.QRectF(pg.Point(x0, lo), pg.Point(x1, hi)), padding=0)

    def suggestPadding(self, axis):
        return 0



class FinPlotItem(pg.GraphicsObject):
    def __init__(self, datasrc, bull_color, bear_color):
        super().__init__()
        self.datasrc = datasrc
        self.bull_color = bull_color
        self.bear_color = bear_color
        self.generatePicture()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())



class CandlestickItem(FinPlotItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = self.datasrc.period / 3
        p.setPen(pg.mkPen(self.bear_color))
        p.setBrush(pg.mkBrush(self.bear_color))
        for t,open,close,high,low in self.datasrc.bear_rows():
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawRect(QtCore.QRectF(t-w, open, w*2, close-open))
        p.setPen(pg.mkPen(self.bull_color))
        p.setBrush(pg.mkBrush(self.bull_color))
        for t,open,close,high,low in self.datasrc.bull_rows():
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawRect(QtCore.QRectF(t-w, open, w*2, close-open))
        p.end()



class VolumeItem(FinPlotItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = self.datasrc.period * 0.7
        p.setPen(pg.mkPen(self.bear_color))
        p.setBrush(pg.mkBrush(self.bear_color))
        for t,open,close,volume in self.datasrc.bear_rows():
            p.drawRect(QtCore.QRectF(t-w, 0, w, volume))
        p.setPen(pg.mkPen(self.bull_color))
        p.setBrush(pg.mkBrush(self.bull_color))
        for t,open,close,volume in self.datasrc.bull_rows():
            p.drawRect(QtCore.QRectF(t-w, 0, w, volume))
        p.end()



def create_plot(title=None, rows=1, maximize=True):
    win = pg.GraphicsWindow(title=title)
    if maximize:
        win.showMaximized()
    # normally first graph is of higher significance, so enlarge
    win.ci.layout.setRowStretchFactor(0, 3)
    axs = []
    for n in range(rows):
        viewbox = FinViewBox(win)
        axs += [add_timestamp_plot(win, viewbox, n)]
    return axs


def add_timestamp_plot(win, viewbox, n):
    global prev_ax
    if prev_ax is not None:
        prev_ax.hideAxis('bottom') # hide the whole previous axis
        win.nextRow()
    ax = pg.PlotItem(viewBox=viewbox, axisItems={'bottom': EpochAxisItem(orientation='bottom')}, name='plot-%i'%n)
    ax.axes['left']['item'].setZValue(10) # put axis in front instead of behind data
    ax.axes['bottom']['item'].setZValue(10)
    prev_ax = ax
    if n:
        ax.setXLink('plot-0')
    if n%2:
        viewbox.setBackgroundColor((240,240,240))
    viewbox.setParent(ax)
    win.addItem(ax)
    return ax


def set_y_range(ax, ymin, ymax):
    ax.vb.auto_scale_y = False
    ax.setYRange(ymin, ymax, padding=0)


def update_datasrc(ax, datasrc):
    viewbox = ax.vb
    if viewbox.datasrc is None:
        viewbox.set_datasrc(datasrc) # for mwheel zoom-scaling
        x0 = datasrc.get_time(1e20, period=-0.5)
        x1 = datasrc.get_time(0, period=+0.5)
        ax.setLimits(xMin=x0, xMax=x1)
    else:
        viewbox.datasrc.addcols(datasrc.df)
        viewbox.datasrc.skip_scale_colcnt = max(viewbox.datasrc.skip_scale_colcnt, datasrc.skip_scale_colcnt)
        viewbox.set_datasrc(viewbox.datasrc) # update zoom


def candlestick_ochl(datasrc, bull_color='#44bb55', bear_color='#dd6666', ax=None):
    if ax is None:
        ax = create_plot(maximize=False)
    datasrc.skip_scale_colcnt = 3 # skip open+close for scaling
    update_datasrc(ax, datasrc)
    item = CandlestickItem(datasrc=datasrc, bull_color=bull_color, bear_color=bear_color)
    ax.addItem(item)


def volume_ocv(datasrc, bull_color='#44bb55', bear_color='#dd6666', ax=None):
    if ax is None:
        ax = create_plot(maximize=False)
    datasrc.skip_scale_colcnt = 3 # skip open+close for scaling
    update_datasrc(ax, datasrc)
    bg = VolumeItem(datasrc=datasrc, bull_color=bull_color, bear_color=bear_color)
    ax.addItem(bg)


def plot(x, y, color='#000000', ax=None, style=None, legend=None):
    if ax is None:
        ax = create_plot(maximize=False)
    datasrc = PandasDataSource(pd.concat([x,y], axis=1))
    update_datasrc(ax, datasrc)
    if legend is not None and ax.legend is None:
        ax.legend = FinLegendItem(border_color=legend_border_color, fill_color=legend_fill_color, size=None, offset=(3,2))
        ax.legend.setParentItem(ax.vb)
    if style is None or style=='-':
        ax.plot(datasrc.x, datasrc.y, pen=pg.mkPen(color), name=legend)
    else:
        symbol = {'v':'t', '^':'t1', '>':'t2', '<':'t3'}.get(style, style) # translate some similar styles
        ax.plot(datasrc.x, datasrc.y, pen=None, symbol=symbol, symbolPen=None, symbolSize=10, symbolBrush=pg.mkBrush(color), name=legend)
    if ax.legend is not None:
        for _,label in ax.legend.items:
            label.setText(label.text, color=legend_text_color)


def _pdtime2epoch(t):
    if type(t) is pd.Series and type(t.iloc[0]) is pd.Timestamp:
        return t.astype('int64') // int(1e9)
    return t


def show():
    QtGui.QApplication.instance().exec_()


# default to black-on-white
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
