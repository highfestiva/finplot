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
from functools import partial
from math import log10, floor, fmod
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


legend_border_color = '#000000dd'
legend_fill_color   = '#00000088'
legend_text_color   = '#dddddd66'
plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
odd_plot_background = '#f0f0f0'
band_color = '#aabbdd'
cross_hair_color = '#000000aa'
draw_color = '#000000'
draw_done_color = '#555555'
significant_digits = 8
v_zoom_padding = 0.02 # padded on top+bottom of plot

windows = [] # no gc
timers = [] # no gc
plotdf2df = {} # for pandas df.plot
epoch_period2 = 0.5



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
        self.col_data_offset = 0 # no. of preceeding columns for other plots
        self.skip_scale_colcnt = 1 # skip at least time for hi/lo, for candle sticks+volume we also skip open and close
        self.cache_hilo_query = ''
        self.cache_hilo_answer = None
        self.scale_colcnt = None # scale on all columns by default
        self.renames = {}
        global epoch_period2
        epoch_period2 = self.period / 2

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
        ycol = self.df.columns[1+self.col_data_offset]
        return self.df[ycol]

    def closest_time(self, t):
        t0,_,_,_,_ = self._hilo(t, t+self.period)
        return t0

    def addcols(self, datasrc):
        orig_col_data_cnt = len(self.df.columns)-1
        newcols = datasrc.df[datasrc.df.columns[1:]] # skip timecol
        cols = list(newcols.columns)
        for i,col in enumerate(cols):
            old_col = col
            while col in self.df.columns:
                cols[i] = col = col+'+'
            if old_col != col:
                datasrc.renames[old_col] = col
        newcols.columns = cols
        self.df = pd.concat([self.df, newcols], axis=1)
        self.skip_scale_colcnt = max(self.skip_scale_colcnt, datasrc.skip_scale_colcnt)
        if datasrc.scale_colcnt:
            self.set_last_scale_columns(True)
        datasrc.df = self.df # they are the same now
        datasrc.col_data_offset = orig_col_data_cnt

    def update(self, datasrc):
        orig_cols = list(self.df.columns)
        timecol = orig_cols[0]
        df = self.df.set_index(timecol)
        data = datasrc.df.set_index(timecol)
        data.columns = [self.renames.get(col, col) for col in data.columns]
        for col in df.columns:
            if col not in data.columns:
                data[col] = df[col]
        data = data.reset_index()
        self.df = data[orig_cols]

    def set_last_scale_columns(self, is_last_scale):
        if is_last_scale:
            self.scale_colcnt = len(self.df.columns)

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
        query = '%.9g,%.9g' % (x0,x1)
        if query != self.cache_hilo_query:
            self.cache_hilo_query = query
            self.cache_hilo_answer = self._hilo(x0, x1)
        return self.cache_hilo_answer

    def _hilo(self, x0, x1):
        df = self.df
        timecol = df.columns[0]
        df = df.loc[((df[timecol]>=x0)&(df[timecol]<=x1)), :]
        if not len(df):
            return 0,0,0,0,0
        t0 = df[timecol].iloc[0]
        t1 = df[timecol].iloc[-1]
        valcols = df.columns[self.skip_scale_colcnt+self.col_data_offset:self.scale_colcnt]
        hi = df[valcols].max().max()
        lo = df[valcols].min().min()
        pad = (hi-lo) * v_zoom_padding
        pad = max(pad, 2e-7) # some very weird bug where too small scale stops rendering
        hi = min(hi+pad, +1e10)
        lo = max(lo-pad, -1e10)
        return t0,t1,hi,lo,len(df)

    def bear_rows(self, colcnt, x0, x1):
        df = self.df
        timecol = df.columns[0]
        opencol = df.columns[1+self.col_data_offset]
        closecol = df.columns[2+self.col_data_offset]
        in_timerange = (df[timecol]>=x0) & (df[timecol]<=x1)
        is_down = df[opencol] > df[closecol] # open higher than close = goes down
        df = df.loc[in_timerange&is_down]
        return self._rows(df, colcnt)

    def bull_rows(self, colcnt, x0, x1):
        df = self.df
        timecol = df.columns[0]
        opencol = df.columns[1+self.col_data_offset]
        closecol = df.columns[2+self.col_data_offset]
        in_timerange = (df[timecol]>=x0) & (df[timecol]<=x1)
        is_up = df[opencol] <= df[closecol] # open lower than close = goes up
        df = df.loc[in_timerange&is_up]
        return self._rows(df, colcnt)

    def _rows(self, df, colcnt):
        if len(df) > 2000:
            df = df.iloc[::len(df)//2000]
        colcnt -= 1 # time is always implied
        cols = [df.columns[0]] + list(df.columns[1+self.col_data_offset:1+self.col_data_offset+colcnt])
        return zip(*[df[c] for c in cols])



class PlotDf(object):
    '''This class is for allowing you to do df.plot(...), as you normally would in Pandas.'''
    def __init__(self, df):
        global plotdf2df
        plotdf2df[self] = df
    def __getattribute__(self, name):
        if name == 'plot':
            return partial(dfplot, plotdf2df[self])
        return getattr(plotdf2df[self], name)
    def __getitem__(self, i):
        return plotdf2df[self].__getitem__(i)
    def __setitem__(self, i, v):
        return plotdf2df[self].__setitem__(i, v)



class FinCrossHair:
    def __init__(self, ax, color):
        self.ax = ax
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=color)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=color)
        self.xtext = pg.TextItem(color=color, anchor=(0,1))
        self.ytext = pg.TextItem(color=color, anchor=(0,0))
        self.vline.setZValue(5)
        self.hline.setZValue(5)
        self.xtext.setZValue(5)
        self.ytext.setZValue(5)
        ax.addItem(self.vline, ignoreBounds=True)
        ax.addItem(self.hline, ignoreBounds=True)
        ax.addItem(self.xtext, ignoreBounds=True)
        ax.addItem(self.ytext, ignoreBounds=True)

    def update(self, point):
        x = point.x() - fmod(point.x()-epoch_period2, epoch_period2*2) + epoch_period2
        self.vline.setPos(x)
        self.hline.setPos(point.y())
        self.xtext.setPos(point)
        self.ytext.setPos(point)
        space = '      '
        self.xtext.setText(space + _epoch2local(x))
        value = _round_to_significant(point.y(), significant_digits)
        self.ytext.setText(space + value)



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



class FinPolyLine(pg.PolyLineROI):
    def __init__(self, vb, *args, **kwargs):
        self.vb = vb # init before parent constructor
        self.texts = []
        super().__init__(*args, **kwargs)

    def addSegment(self, h1, h2, index=None):
        super().addSegment(h1, h2, index)
        text = pg.TextItem(color=draw_color)
        text.segment = self.segments[-1 if index is None else index]
        if index is None:
            self.texts.append(text)
        else:
            self.texts.insert(index, text)
        self.update_text(text)
        self.vb.addItem(text, ignoreBounds=True)

    def removeSegment(self, seg):
        super().removeSegment(seg)
        for text in list(self.texts):
            if text.segment == seg:
                self.vb.removeItem(text)
                self.texts.remove(text)

    def update_text(self, text):
        h0 = text.segment.handles[0]['item']
        h1 = text.segment.handles[1]['item']
        diff = h1.pos() - h0.pos()
        if diff.y() < 0:
            text.setAnchor((0.5,0))
        else:
            text.setAnchor((0.5,1))
        text.setPos(h1.pos())
        text.setText(_draw_line_segment_text(self, text.segment, h0.pos(), h1.pos()))

    def update_texts(self):
        for text in self.texts:
            self.update_text(text)

    def movePoint(self, handle, pos, modifiers=QtCore.Qt.KeyboardModifier(), finish=True, coords='parent'):
        super().movePoint(handle, pos, modifiers, finish, coords)
        self.update_texts()



class FinViewBox(pg.ViewBox):
    def __init__(self, win, init_steps=300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win = win
        self.force_range_update = 0
        self.lines = []
        self.draw_line = None
        self.drawing = False
        self.set_datasrc(None)
        self.setMouseEnabled(x=True, y=False)
        self.init_steps = init_steps

    def set_datasrc(self, datasrc):
        self.datasrc = datasrc
        if not self.datasrc:
            return
        datasrc.init_x0 = datasrc.get_time(offset_from_end=self.init_steps, period=-0.5)
        datasrc.init_x1 = datasrc.get_time(offset_from_end=0, period=+0.5)
        x0,x1,hi,lo,cnt = self.datasrc.hilo(datasrc.init_x0, datasrc.init_x1)
        if cnt >= 20:
            self.set_range(x0, lo, x1, hi, pad=True)

    def wheelEvent(self, ev, axis=None):
        scale_fact = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        vr = self.targetRect()
        center = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(ev.pos()))
        if (center.x()-vr.left())/vr.width() < 0.05: # zoom to far left => all the way left
            center = pg.Point(vr.left(), center.y())
        elif (center.x()-vr.left())/vr.width() > 0.95: # zoom to far right => all the way right
            center = pg.Point(vr.right(), center.y())
        self.zoom_rect(vr, scale_fact, center)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.LeftButton:
            super().mouseDragEvent(ev, axis)
            if ev.isFinish():
                self.force_range_update = 6 # as many as plots, or some more is fine too
                self.update_range()
            return
        if self.draw_line and not self.drawing:
            self.set_draw_line_color(draw_done_color)
        p0 = ev.lastPos()
        p1 = ev.pos()
        p0 = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(p0))
        p1 = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(p1))
        if not self.drawing:
            # add new line
            self.draw_line = FinPolyLine(self, [p0, p1], closed=False, pen=pg.mkPen(draw_color), movable=False)
            self.lines.append(self.draw_line)
            self.addItem(self.draw_line)
            self.drawing = True
        else:
            # draw placed point at end of poly-line
            self.draw_line.movePoint(-1, p1)
        if ev.isFinish():
            self.drawing = False
        ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or not self.draw_line:
            return super().mouseClickEvent(ev)
        # add another segment to the currently drawn line
        p = ev.pos()
        p = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(p))
        self.append_draw_segment(p)
        self.drawing = False
        ev.accept()

    def keyPressEvent(self, ev):
        if ev.text() in ('\r', ' ', '\x1b'): # enter, space, esc
            self.set_draw_line_color(draw_done_color)
            self.draw_line = None
            ev.accept()
        elif ev.text() in ('\x7f', '\b'): # del, backspace
            if self.lines:
                h = self.lines[-1].handles[-1]['item']
                self.lines[-1].removeHandle(h)
                if not self.lines[-1].segments:
                    self.removeItem(self.lines[-1])
                    self.lines = self.lines[:-1]
                    self.draw_line = None
                if self.lines:
                    self.draw_line = self.lines[-1]
                    self.set_draw_line_color(draw_color)
        else:
            super().keyPressEvent(ev)

    def linkedViewChanged(self, view, axis):
        if view:
            tr = self.targetRect()
            vr = view.viewRect()
            period = self.datasrc.period
            is_dirty = view.force_range_update > 0
            if is_dirty or abs(vr.left()-tr.left()) >= period or abs(vr.right()-tr.right()) >= period:
                if is_dirty:
                    view.force_range_update -= 1
                x0,x1,hi,lo,cnt = self.datasrc.hilo(vr.left(), vr.right())
                self.set_range(vr.left(), lo, vr.right(), hi)

    def zoom_rect(self, vr, scale_fact, center):
        if not self.datasrc:
            return
        x_ = vr.left()
        x0 = center.x() + (vr.left()-center.x()) * scale_fact
        x1 = center.x() + (vr.right()-center.x()) * scale_fact
        self.update_range(x0, x1)

    def update_range(self, x0=None, x1=None):
        if x0 is None or x1 is None:
            tr = self.targetRect()
            x0 = tr.left()
            x1 = tr.right()
        x0,x1,hi,lo,cnt = self.datasrc.hilo(x0, x1)
        if cnt < 20:
            return
        self.set_range(x0, lo, x1, hi, pad=True)

    def set_range(self, x0, y0, x1, y1, pad=False):
        if np.isnan(y0) or np.isnan(y1):
            return
        if pad:
            x0 -= self.datasrc.period*0.5
            x1 += self.datasrc.period*0.5
        self.setRange(QtCore.QRectF(pg.Point(x0, y0), pg.Point(x1, y1)), padding=0)

    def append_draw_segment(self, p):
        h0 = self.draw_line.handles[-1]['item']
        h1 = self.draw_line.addFreeHandle(p)
        self.draw_line.addSegment(h0, h1)
        self.drawing = True

    def set_draw_line_color(self, color):
        if self.draw_line:
            pen = pg.mkPen(color)
            for segment in self.draw_line.segments:
                segment.currentPen = segment.pen = pen
                segment.update()

    def suggestPadding(self, axis):
        return 0



class FinPlotItem(pg.GraphicsObject):
    def __init__(self, datasrc, bull_color, bear_color):
        super().__init__()
        self.datasrc = datasrc
        self.bull_color = bull_color
        self.bear_color = bear_color
        self.picture = QtGui.QPicture()
        self.painter = QtGui.QPainter()
        self.dirty = True
        self.lowres_item = None
        # generate picture
        visibleRect = QtCore.QRectF(self.datasrc.init_x0, 0, self.datasrc.init_x1-self.datasrc.init_x0, 0)
        self._generatePicture(visibleRect)

    def paint(self, p, *args):
        viewRect = self.viewRect()
        self.updateDirtyPicture(viewRect)
        p.drawPicture(0, 0, self.picture)

    def updateDirtyPicture(self, visibleRect):
        if self.dirty or \
            visibleRect.left() <= self.cachedRect.left() or \
            visibleRect.right() >= self.cachedRect.right() or \
            visibleRect.width() < self.cachedRect.width() / 10: # optimize when zooming in
            self._generatePicture(visibleRect)

    def _generatePicture(self, boundingRect):
        w = boundingRect.width()
        self.cachedRect = QtCore.QRectF(boundingRect.left()-w, 0, 3*w, 0)
        self.generatePicture(self.cachedRect)
        self.dirty = False

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())



class CandlestickItem(FinPlotItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generatePicture(self, boundingRect):
        w = self.datasrc.period * 0.7
        w2 = w * 0.5
        left,right = boundingRect.left(), boundingRect.right()
        p = self.painter
        p.begin(self.picture)
        p.setPen(pg.mkPen(self.bear_color))
        p.setBrush(pg.mkBrush(self.bear_color))
        rows = 0
        for t,open,close,high,low in self.datasrc.bear_rows(5, left, right):
            if high > low:
                p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawRect(QtCore.QRectF(t-w2, open, w, close-open))
            rows += 1
        p.setPen(pg.mkPen(self.bull_color))
        p.setBrush(pg.mkBrush(self.bull_color))
        for t,open,close,high,low in self.datasrc.bull_rows(5, left, right):
            if high > low:
                p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawRect(QtCore.QRectF(t-w2, open, w, close-open))
        p.end()



class VolumeItem(FinPlotItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generatePicture(self, boundingRect):
        w = self.datasrc.period * 0.7
        w2 = w * 0.5
        left,right = boundingRect.left(), boundingRect.right()
        p = self.painter
        p.begin(self.picture)
        p.setPen(pg.mkPen(self.bear_color))
        p.setBrush(pg.mkBrush(self.bear_color))
        for t,open,close,volume in self.datasrc.bear_rows(4, left, right):
            p.drawRect(QtCore.QRectF(t-w2, 0, w, volume))
        p.setPen(pg.mkPen(self.bull_color))
        p.setBrush(pg.mkBrush(self.bull_color))
        for t,open,close,volume in self.datasrc.bull_rows(4, left, right):
            p.drawRect(QtCore.QRectF(t-w2, 0, w, volume))
        p.end()



def create_plot(title=None, rows=1, init_zoom_periods=1e10, maximize=True):
    global windows
    win = pg.GraphicsWindow(title=title)
    windows.append(win)
    if maximize:
        win.showMaximized()
    win.ci.setContentsMargins(0, 0, 0 ,0)
    # normally first graph is of higher significance, so enlarge
    win.ci.layout.setRowStretchFactor(0, 3)
    axs = []
    prev_ax = None
    for n in range(rows):
        viewbox = FinViewBox(win, init_steps=init_zoom_periods)
        ax = prev_ax = _add_timestamp_plot(win, prev_ax, viewbox, n)
        axs += [ax]
    win.proxy_mmove = pg.SignalProxy(win.scene().sigMouseMoved, rateLimit=60, slot=partial(_mouse_moved, win))
    if len(axs) == 1:
        return axs[0]
    return axs


def candlestick_ochl(datasrc, bull_color='#44bb55', bear_color='#dd6666', ax=None, is_last_scale=True):
    '''The is_last_scale parameter means that no other graphs added afterwards will be included in the
       zoom/pan Y-scaling. Normally the candlesticks provide the relevant scale for this plot area.'''
    if ax is None:
        ax = create_plot(maximize=False)
    datasrc.skip_scale_colcnt = 3 # skip open+close for scaling
    _set_datasrc(ax, datasrc, is_last_scale=is_last_scale)
    item = CandlestickItem(datasrc=datasrc, bull_color=bull_color, bear_color=bear_color)
    item.update_datasrc = partial(_update_datasrc, item)
    ax.addItem(item)
    _set_plot_x_axis_leader(ax)
    return item


def volume_ocv(datasrc, bull_color='#44bb55', bear_color='#dd6666', ax=None, is_last_scale=True):
    if ax is None:
        ax = create_plot(maximize=False)
    datasrc.skip_scale_colcnt = 3 # skip open+close for scaling
    _set_datasrc(ax, datasrc, is_last_scale=is_last_scale)
    item = VolumeItem(datasrc=datasrc, bull_color=bull_color, bear_color=bear_color)
    item.update_datasrc = partial(_update_datasrc, item)
    ax.addItem(item)
    _set_plot_x_axis_leader(ax)
    return item


def plot(x, y, color=None, ax=None, style=None, legend=None, is_last_scale=False):
    datasrc = PandasDataSource(pd.concat([x,y], axis=1))
    return plot_datasrc(datasrc, color=color, ax=ax, style=style, legend=legend, is_last_scale=is_last_scale)


def plot_datasrc(datasrc, color=None, ax=None, style=None, legend=None, is_last_scale=False):
    if ax is None:
        ax = create_plot(maximize=False)
    color = color if color else _get_color(ax)
    _set_datasrc(ax, datasrc, is_last_scale=is_last_scale)
    if legend is not None and ax.legend is None:
        ax.legend = FinLegendItem(border_color=legend_border_color, fill_color=legend_fill_color, size=None, offset=(3,2))
        ax.legend.setParentItem(ax.vb)
    if style is None or style=='-':
        item = ax.plot(datasrc.x, datasrc.y, pen=pg.mkPen(color), name=legend)
    else:
        symbol = {'v':'t', '^':'t1', '>':'t2', '<':'t3'}.get(style, style) # translate some similar styles
        item = ax.plot(datasrc.x, datasrc.y, pen=None, symbol=symbol, symbolPen=None, symbolSize=10, symbolBrush=pg.mkBrush(color), name=legend)
    item.datasrc = datasrc
    item.update_datasrc = partial(_update_datasrc, item)
    if ax.legend is not None:
        for _,label in ax.legend.items:
            label.setText(label.text, color=legend_text_color)
    _set_plot_x_axis_leader(ax)
    return item


def dfplot(df, x=None, y=None, color=None, ax=None, style=None, legend=None):
    legend = legend if legend else y
    return plot(df[x], df[y], color=color, ax=ax, style=style, legend=legend, is_last_scale=True)


def set_y_range(ax, ymin, ymax):
    ax.setLimits(yMin=ymin, yMax=ymax)


def add_band(ax, y0, y1, color=band_color):
    ax.vb.setBackgroundColor(None)
    lr = pg.LinearRegionItem([y0,y1], orientation=pg.LinearRegionItem.Horizontal, brush=pg.mkBrush(color), movable=False)
    lr.setZValue(-10)
    ax.addItem(lr)


def add_time_inspector(ax, inspector):
    win = ax.vb.win
    win.proxy_click = pg.SignalProxy(win.scene().sigMouseClicked, slot=partial(_time_clicked, ax, inspector))


def timer_callback(update_func, seconds, single_shot=False):
    global timers
    timer = QtCore.QTimer()
    timer.timeout.connect(update_func)
    if single_shot:
        timer.setSingleShot(True)
    timer.start(seconds*1000)
    timers.append(timer)


def show():
    if windows:
        QtGui.QApplication.instance().exec_()
        windows.clear()




#################### INTERNALS ####################


def _add_timestamp_plot(win, prev_ax, viewbox, n):
    if prev_ax is not None:
        prev_ax.hideAxis('bottom') # hide the whole previous axis
        win.nextRow()
    ax = pg.PlotItem(viewBox=viewbox, axisItems={'bottom': EpochAxisItem(orientation='bottom')}, name='plot-%i'%n)
    ax.axes['left']['item'].textWidth = 51 # this is to put all graphs on equal footing when texts vary from 0.4 to 2000000
    ax.axes['left']['item'].setStyle(tickLength=-5) # some bug, totally unexplicable (why setting the default value again would fix repaint width as axis scale down)
    ax.axes['left']['item'].setZValue(10) # put axis in front instead of behind data
    ax.axes['bottom']['item'].setZValue(10)
    ax.crosshair = FinCrossHair(ax, color=cross_hair_color)
    if n%2:
        viewbox.setBackgroundColor(odd_plot_background)
    viewbox.setParent(ax)
    win.addItem(ax)
    return ax


def _set_datasrc(ax, datasrc, is_last_scale=False):
    datasrc.set_last_scale_columns(is_last_scale)
    viewbox = ax.vb
    if viewbox.datasrc is None:
        viewbox.set_datasrc(datasrc) # for mwheel zoom-scaling
        _set_x_limits(ax, datasrc)
    else:
        viewbox.datasrc.addcols(datasrc)
        viewbox.set_datasrc(viewbox.datasrc) # update zoom
        datasrc.init_x0 = viewbox.datasrc.init_x0
        datasrc.init_x1 = viewbox.datasrc.init_x1


def _update_datasrc(item, ds):
    item.datasrc.update(ds)
    if isinstance(item, FinPlotItem):
        item.dirty = True
    else:
        item.setData(item.datasrc.x, item.datasrc.y)
    for ax in _axs_with_datasrc(item.datasrc):
        _,x1 = _set_x_limits(ax, item.datasrc)
        tr = ax.vb.targetRect()
        if tr.right() >= x1-item.datasrc.period*3:
            x0 = x1 - tr.width()
            x0,x1,y0,y1,cnt = item.datasrc.hilo(x0, x1)
            ax.vb.set_range(x0, y0, x1, y1, pad=True)
            ax.vb.update()


def _set_plot_x_axis_leader(ax):
    '''The first plot to add some data is the leader. All other's X-axis will follow this one.'''
    if ax.vb.linkedView(0):
        return
    for ax_ in ax.vb.win.ci.items:
        if ax_.vb.name != ax.vb.name:
            ax_.setXLink(ax.vb.name)


def _set_x_limits(ax, datasrc):
    x0 = datasrc.get_time(1e20, period=-0.5)
    x1 = datasrc.get_time(0, period=+0.5)
    ax.setLimits(xMin=x0, xMax=x1)
    return x0, x1


def _items_with_datasrc(datasrc):
    for ax in axs_with_datasrc(datasrc):
        for item in ax.items:
            if item.datasrc == datasrc:
                yield item


def _axs_with_datasrc(datasrc):
    for win in windows:
        for ax in win.ci.items:
            if ax.vb.datasrc == datasrc:
                yield ax


def _mouse_moved(win, ev):
    pos = ev[0]
    for ax in win.ci.items:
        point = ax.vb.mapSceneToView(pos)
        ax.crosshair.update(point)


def _time_clicked(ax, inspector, ev):
    pos = ev[0].scenePos()
    point = ax.vb.mapSceneToView(pos)
    t = point.x() - epoch_period2
    t = ax.vb.datasrc.closest_time(t)
    inspector(t, point.y())


def _get_color(ax):
    index = len(ax.items) - 4
    return plot_colors[index%len(plot_colors)]


def _pdtime2epoch(t):
    if isinstance(t, pd.Series) and isinstance(t.iloc[0], pd.Timestamp):
        return t.astype('int64') // int(1e9)
    return t


def _epoch2local(t):
    return datetime.fromtimestamp(t).isoformat().replace('T',' ').rsplit(':',1)[0]


def _round_to_significant(x, num_significant):
    x = round(x, num_significant-1-int(floor(log10(abs(x)))))
    fmt = '%%.%if' % num_significant
    return (fmt % x)[:num_significant+2]


def _draw_line_segment_text(polyline, segment, pos0, pos1):
        diff = pos1 - pos0
        mins = int(abs(diff.x()) / 60)
        hours = mins//60
        mins = mins%60
        ts = '%0.2i:%0.2i' % (hours, mins)
        percent = '%+.2f' % (100 * pos1.y() / pos0.y() - 100)
        extra = _draw_line_extra_text(polyline, segment, pos0, pos1)
        return '%s %% %s (%s)' % (percent, extra, ts)


def _draw_line_extra_text(polyline, segment, pos0, pos1):
    '''Shows the proportions of this line height compared to the previous segment.'''
    prev_text = None
    for text in polyline.texts:
        if prev_text is not None and text.segment == segment:
            h0 = prev_text.segment.handles[0]['item']
            h1 = prev_text.segment.handles[1]['item']
            prev_change = h1.pos().y() / h0.pos().y() - 1
            if not abs(prev_change) > 1e-8:
                break
            this_change = pos1.y() / pos0.y() - 1
            change_part = abs(this_change / prev_change)
            return ' = 1:%.2f ' % change_part
        prev_text = text
    return ''


# default to black-on-white
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
