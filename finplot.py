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
odd_plot_background = '#f0f0f0'
band_color = '#aabbdd'
cross_hair_color = '#000000aa'
draw_color = '#000000'
draw_done_color = '#555555'
significant_digits = 8
v_zoom_padding = 0.02 # padded on top+bottom of plot

windows = [] # disallow garbage collecting
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
        self.skip_scale_colcnt = 1 # skip at least time for hi/lo, for candle sticks+volume we also skip open and close
        self.cache_hilo_query = ''
        self.cache_hilo_answer = None
        self.scale_colcnt = None # scale on all columns by default
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
        ycol = self.df.columns[1]
        return self.df[ycol]

    def closest_time(self, t):
        t0,_,_,_,_ = self._hilo(t, t+self.period)
        return t0

    def addcols(self, datasrc):
        newcols = datasrc.df[datasrc.df.columns[1:]] # skip timecol
        self.df = pd.concat([self.df, newcols], axis=1)
        self.skip_scale_colcnt = max(self.skip_scale_colcnt, datasrc.skip_scale_colcnt)
        if datasrc.scale_colcnt:
            self.set_last_scale_columns(True)

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
        valcols = df.columns[self.skip_scale_colcnt:self.scale_colcnt]
        hi = df[valcols].max().max()
        lo = df[valcols].min().min()
        pad = (hi-lo) * v_zoom_padding
        return t0,t1,hi+pad,lo-pad,len(df)

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
        self.lines = []
        self.draw_line = None
        self.drawing = False
        self.set_datasrc(None)
        self.setMouseEnabled(x=True, y=False)
        self.init_steps = init_steps
        self.heavies = []
        self.heavies_blind_cnt = 50
        self.heavies_timer = QtCore.QTimer()
        self.heavies_timer.timeout.connect(self.show_heavies)
        self.heavies_timer.start(50)

    def set_datasrc(self, datasrc):
        self.datasrc = datasrc
        if not self.datasrc:
            return
        x0 = datasrc.get_time(offset_from_end=self.init_steps, period=-0.5)
        x1 = datasrc.get_time(offset_from_end=0, period=+0.5)
        t0,t1,hi,lo,cnt = self.datasrc.hilo(x0, x1)
        if cnt >= 20:
            self._setRange(t0, lo, t1, hi)

    def add_heavy_item(self, item):
        item.setVisible(False)
        self.heavies.append(item)
        self.heavies_blind_cnt = 50

    def wheelEvent(self, ev, axis=None):
        scale_fact = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        vr = self.targetRect()
        center = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(ev.pos()))
        if (center.x()-vr.left())/vr.width() < 0.05: # zoom to far left => all the way left
            center = pg.Point(vr.left(), center.y())
        elif (center.x()-vr.left())/vr.width() > 0.95: # zoom to far right => all the way right
            center = pg.Point(vr.right(), center.y())
        self.scaleRect(vr, scale_fact, center)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.LeftButton:
            return super().mouseDragEvent(ev, axis)
        p0 = ev.lastPos()
        p1 = ev.pos()
        p0 = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(p0))
        p1 = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(p1))
        if self.draw_line is None:
            # add new line
            self.draw_line = FinPolyLine(self, [p0, p1], closed=False, pen=pg.mkPen(draw_color), movable=False)
            self.lines.append(self.draw_line)
            self.addItem(self.draw_line)
            self.drawing = True
        elif self.drawing:
            # draw placed point at end of poly-line
            self.draw_line.movePoint(-1, p1)
        else:
            self.append_draw_segment(p1)
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
        tr = self.targetRect()
        vr = view.viewRect() if view else tr
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
        self._setRange(x0, lo, x1, hi)

    def _setRange(self, x0, y0, x1, y1):
        if np.isnan(y0) or np.isnan(y1):
            return
        for item in self.heavies:
            item.setVisible(False) # deferred rendering for zoom+pan performance
        self.setRange(QtCore.QRectF(pg.Point(x0, y0), pg.Point(x1, y1)), padding=0)
        self.heavies_blind_cnt = 2 # unblind in this many ticks

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

    def show_heavies(self):
        self.heavies_blind_cnt -= 1
        if self.heavies_blind_cnt != 0:
            return
        for item in self.heavies:
            item.setVisible(True)

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
        w = self.datasrc.period * 0.7
        w2 = w * 0.5
        p.setPen(pg.mkPen(self.bear_color))
        p.setBrush(pg.mkBrush(self.bear_color))
        for t,open,close,high,low in self.datasrc.bear_rows():
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawRect(QtCore.QRectF(t-w2, open, w, close-open))
        p.setPen(pg.mkPen(self.bull_color))
        p.setBrush(pg.mkBrush(self.bull_color))
        for t,open,close,high,low in self.datasrc.bull_rows():
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawRect(QtCore.QRectF(t-w2, open, w, close-open))
        p.end()



class VolumeItem(FinPlotItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = self.datasrc.period * 0.7
        w2 = w * 0.5
        p.setPen(pg.mkPen(self.bear_color))
        p.setBrush(pg.mkBrush(self.bear_color))
        for t,open,close,volume in self.datasrc.bear_rows():
            p.drawRect(QtCore.QRectF(t-w2, 0, w, volume))
        p.setPen(pg.mkPen(self.bull_color))
        p.setBrush(pg.mkBrush(self.bull_color))
        for t,open,close,volume in self.datasrc.bull_rows():
            p.drawRect(QtCore.QRectF(t-w2, 0, w, volume))
        p.end()



def create_plot(title=None, rows=1, init_zoom_periods=300, maximize=True):
    global windows
    win = pg.GraphicsWindow(title=title)
    windows.append(win)
    if maximize:
        win.showMaximized()
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
    _update_datasrc(ax, datasrc, is_last_scale=is_last_scale)
    item = CandlestickItem(datasrc=datasrc, bull_color=bull_color, bear_color=bear_color)
    ax.addItem(item)
    ax.vb.add_heavy_item(item) # heavy = deferred rendering
    _update_main_plot(ax)
    return item


def volume_ocv(datasrc, bull_color='#44bb55', bear_color='#dd6666', ax=None):
    if ax is None:
        ax = create_plot(maximize=False)
    datasrc.skip_scale_colcnt = 3 # skip open+close for scaling
    _update_datasrc(ax, datasrc)
    item = VolumeItem(datasrc=datasrc, bull_color=bull_color, bear_color=bear_color)
    ax.addItem(item)
    ax.vb.add_heavy_item(item) # heavy = deferred rendering
    _update_main_plot(ax)
    return item


def plot(x, y, color='#000000', ax=None, style=None, legend=None):
    datasrc = PandasDataSource(pd.concat([x,y], axis=1))
    return plot_datasrc(datasrc, color=color, ax=ax, style=style, legend=legend)


def plot_datasrc(datasrc, color='#000000', ax=None, style=None, legend=None):
    if ax is None:
        ax = create_plot(maximize=False)
    _update_datasrc(ax, datasrc)
    if legend is not None and ax.legend is None:
        ax.legend = FinLegendItem(border_color=legend_border_color, fill_color=legend_fill_color, size=None, offset=(3,2))
        ax.legend.setParentItem(ax.vb)
    if style is None or style=='-':
        item = ax.plot(datasrc.x, datasrc.y, pen=pg.mkPen(color), name=legend)
    else:
        symbol = {'v':'t', '^':'t1', '>':'t2', '<':'t3'}.get(style, style) # translate some similar styles
        item = ax.plot(datasrc.x, datasrc.y, pen=None, symbol=symbol, symbolPen=None, symbolSize=10, symbolBrush=pg.mkBrush(color), name=legend)
    if ax.legend is not None:
        for _,label in ax.legend.items:
            label.setText(label.text, color=legend_text_color)
    _update_main_plot(ax)
    return item


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


def show():
    if windows:
        QtGui.QApplication.instance().exec_()




#################### INTERNALS ####################


def _add_timestamp_plot(win, prev_ax, viewbox, n):
    if prev_ax is not None:
        prev_ax.hideAxis('bottom') # hide the whole previous axis
        win.nextRow()
    ax = pg.PlotItem(viewBox=viewbox, axisItems={'bottom': EpochAxisItem(orientation='bottom')}, name='plot-%i'%n)
    ax.axes['left']['item'].setZValue(10) # put axis in front instead of behind data
    ax.axes['bottom']['item'].setZValue(10)
    ax.crosshair = FinCrossHair(ax, color=cross_hair_color)
    if n%2:
        viewbox.setBackgroundColor(odd_plot_background)
    viewbox.setParent(ax)
    win.addItem(ax)
    return ax


def _update_datasrc(ax, datasrc, is_last_scale=False):
    datasrc.set_last_scale_columns(is_last_scale)
    viewbox = ax.vb
    if viewbox.datasrc is None:
        viewbox.set_datasrc(datasrc) # for mwheel zoom-scaling
        x0 = datasrc.get_time(1e20, period=-1.0)
        x1 = datasrc.get_time(0, period=+0.5)
        ax.setLimits(xMin=x0, xMax=x1)
    else:
        viewbox.datasrc.addcols(datasrc)
        viewbox.set_datasrc(viewbox.datasrc) # update zoom


def _update_main_plot(ax):
    '''The first plot to add some data is the leader. All other's X-axis will follow this one.'''
    if ax.vb.linkedView(0):
        return
    for ax_ in ax.vb.win.ci.items:
        if ax_.vb.name != ax.vb.name:
            ax_.setXLink(ax.vb.name)


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


def _pdtime2epoch(t):
    if type(t) is pd.Series and type(t.iloc[0]) is pd.Timestamp:
        return t.astype('int64') // int(1e9)
    return t


def _epoch2local(t):
    return datetime.fromtimestamp(t).isoformat().replace('T',' ').rsplit(':',1)[0]


def _round_to_significant(x, num_significant):
    x = round(x, num_significant-1-int(floor(log10(abs(x)))))
    return ('%f' % x)[:num_significant+1]


def _draw_line_segment_text(polyline, segment, pos0, pos1):
        diff = pos1 - pos0
        mins = int(abs(diff.x()) / 60)
        hours = mins//60
        mins = mins%60
        ts = '%0.2i:%0.2i' % (hours, mins)
        percent = '%+.2f' % (100 * pos1.y() / pos0.y() - 100)
        extra = draw_line_extra_text(polyline, segment, pos0, pos1)
        return '%s %% %s (%s)' % (percent, extra, ts)


def draw_line_extra_text(polyline, segment, pos0, pos1):
    '''Overwrite to fill in additional information per drawn line segment.'''
    return ''


# default to black-on-white
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
