# -*- coding: utf-8 -*-
'''
Financial data plotter with better defaults, api, behavior and performance than
mpl_finance and plotly.

Lines up your time-series with a shared X-axis; ideal for volume, RSI, etc.

Zoom does something similar to what you'd normally expect for financial data,
where the Y-axis is auto-scaled to highest high and lowest low in the active
region.
'''

from ast import literal_eval
from collections import OrderedDict, defaultdict
from datetime import datetime
from decimal import Decimal
from functools import partial, partialmethod
from math import ceil, floor, fmod
import numpy as np
import os.path
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui



# appropriate types
ColorMap = pg.ColorMap

# module definitions, mostly colors
legend_border_color = '#777'
legend_fill_color   = '#6668'
legend_text_color   = '#ddd6'
soft_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
hard_colors = ['#000000', '#772211', '#000066', '#555555', '#0022cc', '#ffcc00']
colmap_clash = ColorMap([0.0, 0.2, 0.6, 1.0], [[127, 127, 255, 51], [0, 0, 127, 51], [255, 51, 102, 51], [255, 178, 76, 51]])
foreground = '#000'
background = '#fff'
odd_plot_background = '#fff'
candle_bull_color = '#26a69a'
candle_bear_color = '#ef5350'
candle_bull_body_color = background
candle_shadow_width = 1
volume_bull_color = '#92d2cc'
volume_bear_color = '#f7a9a7'
volume_bull_body_color = volume_bull_color
volume_neutral_color = '#bbb'
poc_color = '#006'
band_color = '#d2dfe6'
cross_hair_color = '#0007'
draw_line_color = '#000'
draw_done_color = '#555'
significant_decimals = 8
significant_eps = 1e-8
max_zoom_points = 20 # number of visible candles when maximum zoomed in
top_graph_scale = 2
clamp_grid = True
right_margin_candles = 5 # whitespace at the right-hand side
side_margin = 0.5
lod_candles = 3000
lod_labels = 700
cache_candle_factor = 3 # factor extra candles rendered to buffer
y_pad = 0.03 # 3% padding at top and bottom of autozoom plots
y_label_width = 65
long_time = 2*365*24*60*60*1e9
display_timezone = None  # default to local
winx,winy,winw,winh = 400,300,800,400

app = None
windows = [] # no gc
timers = [] # no gc
sounds = {} # no gc
epoch_period = 1e30
last_ax = None # always assume we want to plot in the last axis, unless explicitly specified
overlay_axs = [] # for keeping track of candlesticks in overlays
viewrestore = False
master_data = {}



lerp = lambda t,a,b: t*b+(1-t)*a



class EpochAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb

    def tickStrings(self, values, scale, spacing):
        if self.mode == 'num':
            return ['%g'%v for v in values]
        conv = _x2year if self.mode=='year' else _x2local_t
        return [conv(self.vb.datasrc, value)[0] for value in values]

    def tickValues(self, minVal, maxVal, size):
        self.mode = 'num'
        ax = self.vb.parent()
        datasrc = _get_datasrc(ax, require=False)
        if datasrc is None or not self.vb.x_indexed:
            return super().tickValues(minVal, maxVal, size)
        # see if we have time
        self.mode = 'time'
        t0,t1,_,_,_ = datasrc.hilo(minVal, maxVal)
        if t1-t0 <= long_time:
            return super().tickValues(minVal, maxVal, size)
        # year index calculation
        self.mode = 'year'
        maxVal = min(datasrc.df.index[-1], maxVal)
        y0 = int(_x2utc(datasrc, minVal)[0][:4])
        y1 = int(_x2utc(datasrc, maxVal)[0][:4])
        step = (y1-y0)//12 or 1
        years = pd.Series(pd.to_datetime(['%s'%y for y in range(y0,y1+1,step)]))
        years_indices = [ceil(yi) for yi in _pdtime2index(ax, years)]
        return [(0,years_indices)]

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs and not self.style['showValues']:
            pen,p0,p1 = specs[0]
            specs = [(_makepen('#fff0'),p0,p1)] + list(specs[1:]) # don't draw axis if hiding values
        return specs



class YAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb
        self.hide_strings = False
        self.style['autoExpandTextSpace'] = False
        self.style['autoReduceTextSpace'] = False

    def tickStrings(self, values, scale, spacing):
        if self.hide_strings:
            return []
        return ['%g'%self.vb.yscale.xform(value) for value in values]



class YScale:
    def __init__(self, scaletype, scalef):
        self.scaletype = scaletype
        self.set_scale(scalef)

    def set_scale(self, scale):
        self.scalef = scale

    def xform(self, y):
        if self.scaletype == 'log':
            y = 10**y
        y = y * self.scalef
        return y

    def invxform(self, y, verify=False):
        y /= self.scalef
        if self.scaletype == 'log':
            if verify and y <= 0:
                return -1e6
            y = np.log10(y)
        return y



class PandasDataSource:
    '''Candle sticks: create with five columns: time, open, close, hi, lo - in that order.
       Volume bars: create with three columns: time, open, close, volume - in that order.
       For all other types, time needs to be first, usually followed by one or more Y-columns.'''
    def __init__(self, df):
        if type(df.index) == pd.DatetimeIndex or df.index[-1]>1e7 or '.RangeIndex' not in str(type(df.index)):
            df = df.reset_index()
        self.df = df.copy()
        # manage time column
        if _has_timecol(self.df):
            timecol = self.df.columns[0]
            dtype = str(df[timecol].dtype)
            isnum = ('int' in dtype or 'float' in dtype) and df[timecol].iloc[-1] < 1e7
            if not isnum:
                self.df[timecol] = _pdtime2epoch(df[timecol])
            self.standalone = _is_standalone(self.df[timecol])
            self.col_data_offset = 1 # no. of preceeding columns for other plots and time column
        else:
            self.standalone = False
            self.col_data_offset = 0 # no. of preceeding columns for other plots and time column
        # setup data for joining data sources and zooming
        self.scale_cols = [i for i in range(self.col_data_offset,len(self.df.columns)) if self.df.iloc[:,i].dtype!=object]
        self.cache_hilo = OrderedDict()
        self.renames = {}
        newcols = []
        for col in self.df.columns:
            oldcol = col
            while col in newcols:
                col = str(col)+'+'
            newcols.append(col)
            if oldcol != col:
                self.renames[oldcol] = col
        self.df.columns = newcols
        self.pre_update = lambda df: df
        self.post_update = lambda df: df
        self._period = None
        self.is_sparse = self.df[self.df.columns[self.col_data_offset]].isnull().sum().max() > len(self.df)//2

    @property
    def period(self):
        if len(self.df) <= 1:
            return 1
        if not self._period:
            timecol = self.df.columns[0]
            self._period = self.df[timecol].diff().median() / 1e9
        return self._period

    @property
    def index(self):
        return self.df.index

    @property
    def x(self):
        timecol = self.df.columns[0]
        return self.df[timecol]

    @property
    def y(self):
        col = self.df.columns[self.col_data_offset]
        return self.df[col]

    @property
    def z(self):
        col = self.df.columns[self.col_data_offset+1]
        return self.df[col]

    @property
    def xlen(self):
        return len(self.df)+right_margin_candles

    def calc_significant_decimals(self):
        ser = self.z if len(self.scale_cols)>1 else self.y
        absdiff = ser.diff().abs()
        absdiff[absdiff<1e-30] = 1e30
        num = smallest_diff = absdiff.min()
        for _ in range(2):
            s = '%e' % num
            base,_,exp = s.partition('e')
            base = base.rstrip('0')
            exp = -int(exp)
            max_base_decimals = min(5, -exp+2) if exp < 0 else 3
            base_decimals = max(0, min(max_base_decimals, len(base)-2))
            decimals = exp + base_decimals
            decimals = max(0, min(10, decimals))
            if decimals <= 3:
                break
            # retry with full number, to see if we can get the number of decimals down
            num = smallest_diff + abs(ser.min())
        smallest_diff = max(10**(-decimals), smallest_diff)
        return decimals, smallest_diff

    def update_init_x(self, init_steps):
        self.init_x0, self.init_x1 = _xminmax(self, x_indexed=True, init_steps=init_steps)

    def closest_time(self, x):
        timecol = self.df.columns[0]
        return self.df.loc[int(x), timecol]

    def timebased(self):
        return self.df.iloc[-1,0] > 1e7

    def addcols(self, datasrc):
        new_scale_cols = [c+len(self.df.columns)-datasrc.col_data_offset for c in datasrc.scale_cols]
        self.scale_cols += new_scale_cols
        orig_col_data_cnt = len(self.df.columns)
        if _has_timecol(datasrc.df):
            timecol = self.df.columns[0]
            df = self.df.set_index(timecol)
            timecol = timecol if timecol in datasrc.df.columns else datasrc.df.columns[0]
            newcols = datasrc.df.set_index(timecol)
        else:
            df = self.df
            newcols = datasrc.df
        cols = list(newcols.columns)
        for i,col in enumerate(cols):
            old_col = col
            while col in self.df.columns:
                cols[i] = col = str(col)+'+'
            if old_col != col:
                datasrc.renames[old_col] = col
        newcols.columns = cols
        self.df = pd.concat([df, newcols], axis=1)
        if _has_timecol(datasrc.df):
            self.df.reset_index(inplace=True)
        datasrc.df = self.df # they are the same now
        datasrc.init_x0 = self.init_x0
        datasrc.init_x1 = self.init_x1
        datasrc.col_data_offset = orig_col_data_cnt
        datasrc.scale_cols = new_scale_cols
        self.cache_hilo = OrderedDict()
        self._period = None
        datasrc._period = None
        ldf2 = len(self.df) / 2
        self.is_sparse = self.is_sparse or self.df[self.df.columns[self.col_data_offset]].isnull().sum().max() > ldf2
        datasrc.is_sparse = datasrc.is_sparse or datasrc.df[datasrc.df.columns[datasrc.col_data_offset]].isnull().sum().max() > ldf2

    def update(self, datasrc):
        df = self.pre_update(self.df)
        orig_cols = list(df.columns)
        timecol,orig_cols = orig_cols[0],orig_cols[1:]
        df = df.set_index(timecol)
        input_df = datasrc.df.set_index(datasrc.df.columns[0])
        input_df.columns = [self.renames.get(col, col) for col in input_df.columns]
        # pad index if the input data is a sub-set
        input_df = pd.merge(input_df, df[[]], how='outer', left_index=True, right_index=True)
        for col in df.columns:
            if col not in input_df.columns:
                input_df[col] = df[col]
        input_df = self.post_update(input_df)
        input_df = input_df.reset_index()
        self.df = input_df[[input_df.columns[0]]+orig_cols] if orig_cols else input_df
        self.init_x1 = self.xlen - side_margin
        self.cache_hilo = OrderedDict()
        self._period = None

    def set_df(self, df):
        self.df = df
        self.cache_hilo = OrderedDict()
        self._period = None

    def hilo(self, x0, x1):
        '''Return five values in time range: t0, t1, highest, lowest, number of rows.'''
        if x0 == x1:
            x0 = x1 = int(x1)
        else:
            x0,x1 = int(x0+0.5),int(x1)
        query = '%i,%i' % (x0,x1)
        if query not in self.cache_hilo:
            v = self.cache_hilo[query] = self._hilo(x0, x1)
        else:
            # re-insert to raise prio
            v = self.cache_hilo[query] = self.cache_hilo.pop(query)
        if len(self.cache_hilo) > 100: # drop if too many
            del self.cache_hilo[next(iter(self.cache_hilo))]
        return v

    def _hilo(self, x0, x1):
        df = self.df.loc[x0:x1, :]
        if not len(df):
            return 0,0,0,0,0
        timecol = df.columns[0]
        t0 = df[timecol].iloc[0]
        t1 = df[timecol].iloc[-1]
        valcols = df.columns[self.scale_cols]
        hi = df[valcols].max().max()
        lo = df[valcols].min().min()
        return t0,t1,hi,lo,len(df)

    def rows(self, colcnt, x0, x1, yscale, lod=True):
        df = self.df.loc[x0:x1, :]
        if self.is_sparse:
            df = df.loc[df.iloc[:,self.col_data_offset].notna(), :]
        origlen = len(df)
        return self._rows(df, colcnt, yscale=yscale, lod=lod), origlen

    def _rows(self, df, colcnt, yscale, lod):
        if lod and len(df) > lod_candles:
            df = df.iloc[::len(df)//lod_candles]
        colcnt -= 1 # time is always implied
        colidxs = [0] + list(range(self.col_data_offset, self.col_data_offset+colcnt))
        dfr = df.iloc[:,colidxs]
        if yscale.scaletype == 'log' or yscale.scalef != 1:
            dfr = dfr.copy()
            for i in range(1, colcnt+1):
                if dfr.iloc[:,i].dtype != object:
                    dfr.iloc[:,i] = yscale.invxform(dfr.iloc[:,i])
        return dfr

    def __eq__(self, other):
        return id(self) == id(other) or id(self.df) == id(other.df)

    def __hash__(self):
        return id(self)


class FinWindow(pg.GraphicsLayoutWidget):
    def __init__(self, title, **kwargs):
        global winx, winy
        self.title = title
        pg.mkQApp()
        super().__init__(**kwargs)
        self.setWindowTitle(title)
        self.setGeometry(winx, winy, winw, winh)
        winx += 40
        winy += 40
        self.centralWidget.installEventFilter(self)
        self.ci.setContentsMargins(0, 0, 0, 0)
        self.ci.setSpacing(-1)
        self.closing = False

    @property
    def axs(self):
        return [ax for ax in self.ci.items if isinstance(ax, pg.PlotItem)]

    def close(self):
        self.closing = True
        _savewindata(self)
        _clear_timers()
        return super().close()

    def eventFilter(self, obj, ev):
        if ev.type()== QtCore.QEvent.WindowDeactivate:
            _savewindata(self)
        return False

    def leaveEvent(self, ev):
        if not self.closing:
            super().leaveEvent(ev)


class FinCrossHair:
    def __init__(self, ax, color):
        self.ax = ax
        self.x = 0
        self.y = 0
        self.clamp_x = 0
        self.clamp_y = 0
        self.infos = []
        pen = pg.mkPen(color=color, style=QtCore.Qt.CustomDashLine, dash=[7, 7])
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self.xtext = pg.TextItem(color=color, anchor=(0,1))
        self.ytext = pg.TextItem(color=color, anchor=(0,0))
        self.vline.setZValue(50)
        self.hline.setZValue(50)
        self.xtext.setZValue(50)
        self.ytext.setZValue(50)
        self.show()

    def update(self, point=None):
        if point is not None:
            self.x,self.y = x,y = point.x(),point.y()
        else:
            x,y = self.x,self.y
        x,y = _clamp_xy(self.ax, x,y)
        if x == self.clamp_x and y == self.clamp_y:
            return
        self.clamp_x,self.clamp_y = x,y
        self.vline.setPos(x)
        self.hline.setPos(y)
        self.xtext.setPos(x, y)
        self.ytext.setPos(x, y)
        rng = self.ax.vb.y_max - self.ax.vb.y_min
        rngmax = abs(self.ax.vb.y_min) + rng # any approximation is fine
        sd,se = (self.ax.significant_decimals,self.ax.significant_eps) if clamp_grid else (significant_decimals,significant_eps)
        timebased = False
        if self.ax.vb.x_indexed:
            xtext,timebased = _x2local_t(self.ax.vb.datasrc, x)
        else:
            xtext = _round_to_significant(rng, rngmax, x, sd, se)
        linear_y = y
        y = self.ax.vb.yscale.xform(y)
        ytext = _round_to_significant(rng, rngmax, y, sd, se)
        if not timebased:
            if xtext:
                xtext = 'x ' + xtext
            ytext = 'y ' + ytext
        far_right = self.ax.viewRect().x() + self.ax.viewRect().width()*0.9
        far_bottom = self.ax.viewRect().y() + self.ax.viewRect().height()*0.1
        close2right = x > far_right
        close2bottom = linear_y < far_bottom
        try:
            for info in self.infos:
                xtext,ytext = info(x,y,xtext,ytext)
        except Exception as e:
            print('Crosshair error:', type(e), e)
        space = '      '
        if close2right:
            xtext = xtext + space
            ytext = ytext + space
            xanchor = [1,1]
            yanchor = [1,0]
        else:
            xtext = space + xtext
            ytext = space + ytext
            xanchor = [0,1]
            yanchor = [0,0]
        if close2bottom:
            ytext = ytext + space
            yanchor = [1,1]
            if close2right:
                xanchor = [1,2]
        self.xtext.setAnchor(xanchor)
        self.ytext.setAnchor(yanchor)
        self.xtext.setText(xtext)
        self.ytext.setText(ytext)

    def show(self):
        self.ax.addItem(self.vline, ignoreBounds=True)
        self.ax.addItem(self.hline, ignoreBounds=True)
        self.ax.addItem(self.xtext, ignoreBounds=True)
        self.ax.addItem(self.ytext, ignoreBounds=True)

    def hide(self):
        self.ax.removeItem(self.xtext)
        self.ax.removeItem(self.ytext)
        self.ax.removeItem(self.vline)
        self.ax.removeItem(self.hline)



class FinLegendItem(pg.LegendItem):
    def __init__(self, border_color, fill_color, **kwargs):
        super().__init__(**kwargs)
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(20)
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
        text = pg.TextItem(color=draw_line_color)
        text.setZValue(50)
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

    def segmentClicked(self, segment, ev=None, pos=None):
        pos = segment.mapToParent(ev.pos())
        pos = _clamp_point(self.vb.parent(), pos)
        super().segmentClicked(segment, pos=pos)
        self.update_texts()

    def addHandle(self, info, index=None):
        handle = super().addHandle(info, index)
        handle.movePoint = partial(_roihandle_move_snap, self.vb, handle.movePoint)
        return handle


class FinLine(pg.GraphicsObject):
    def __init__(self, points, pen):
        super().__init__()
        self.points = points
        self.pen = pen

    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawPath(self.shape())

    def shape(self):
        p = QtGui.QPainterPath()
        p.moveTo(*self.points[0])
        p.lineTo(*self.points[1])
        return p

    def boundingRect(self):
        return self.shape().boundingRect()


class FinEllipse(pg.EllipseROI):
    def addRotateHandle(self, *args, **kwargs):
        pass


class FinRect(pg.RectROI):
    def __init__(self, ax, brush, *args, **kwargs):
        self.ax = ax
        self.brush = brush
        super().__init__(*args, **kwargs)

    def paint(self, p, *args):
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)
        p.setBrush(self.brush)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)

    def addScaleHandle(self, *args, **kwargs):
        if self.resizable:
            super().addScaleHandle(*args, **kwargs)


class FinViewBox(pg.ViewBox):
    def __init__(self, win, init_steps=300, yscale=YScale('linear', 1), v_zoom_scale=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win = win
        self.init_steps = init_steps
        self.yscale = yscale
        self.v_zoom_scale = v_zoom_scale
        self.master_viewbox = None
        self.rois = []
        self.isMouseLeftDrag = False
        self.reset()

    def reset(self):
        self.v_zoom_baseline = 0.5
        self.v_autozoom = True
        self.max_zoom_points_f = 1
        self.y_max = 1000
        self.y_min = 0
        self.y_positive = True
        self.x_indexed = True
        self.force_range_update = 0
        while self.rois:
            self.remove_last_roi()
        self.draw_line = None
        self.drawing = False
        self.standalones = set()
        self.updating_linked = False
        self.set_datasrc(None)
        self.setMouseEnabled(x=True, y=False)

    def set_datasrc(self, datasrc):
        self.datasrc = datasrc
        if not self.datasrc:
            return
        datasrc.update_init_x(self.init_steps)

    def pre_process_data(self):
        if self.datasrc and self.datasrc.scale_cols:
            df = self.datasrc.df.iloc[:, self.datasrc.scale_cols]
            self.y_max = df.max().max()
            self.y_min = df.min().min()
            if self.y_min <= 0:
                self.y_positive = False

    @property
    def datasrc_or_standalone(self):
        ds = self.datasrc
        if not ds and self.standalones:
            ds = next(iter(self.standalones))
        return ds

    def wheelEvent(self, ev, axis=None):
        if self.master_viewbox:
            return self.master_viewbox.wheelEvent(ev, axis=axis)
        if ev.modifiers() == QtCore.Qt.ControlModifier:
            scale_fact = 1
            self.v_zoom_scale /= 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        else:
            scale_fact = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        vr = self.targetRect()
        center = self.mapToView(ev.pos())
        if (center.x()-vr.left())/vr.width() < 0.05: # zoom to far left => all the way left
            center = pg.Point(vr.left(), center.y())
        elif (center.x()-vr.left())/vr.width() > 0.95: # zoom to far right => all the way right
            center = pg.Point(vr.right(), center.y())
        self.zoom_rect(vr, scale_fact, center)
        # update crosshair
        _mouse_moved(self.win, None)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        axis = 0 # don't constrain drag direction
        if self.master_viewbox:
            return self.master_viewbox.mouseDragEvent(ev, axis=axis)
        if not self.datasrc:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            self.mouseLeftDrag(ev, axis)
        elif ev.button() == QtCore.Qt.MiddleButton:
            self.mouseMiddleDrag(ev, axis)
        elif ev.button() == QtCore.Qt.RightButton:
            self.mouseRightDrag(ev, axis)
        else:
            super().mouseDragEvent(ev, axis)

    def mouseLeftDrag(self, ev, axis):
        '''Ctrl+LButton draw lines.'''
        if ev.modifiers() != QtCore.Qt.ControlModifier:
            super().mouseDragEvent(ev, axis)
            if ev.isFinish():
                self.isMouseLeftDrag = False
            else:
                self.isMouseLeftDrag = True
            if ev.isFinish() or self.drawing:
                self.refresh_all_y_zoom()
            if not self.drawing:
                return
        if self.draw_line and not self.drawing:
            self.set_draw_line_color(draw_done_color)
        p1 = self.mapToView(ev.pos())
        p1 = _clamp_point(self.parent(), p1)
        if not self.drawing:
            # add new line
            p0 = self.mapToView(ev.lastPos())
            p0 = _clamp_point(self.parent(), p0)
            self.draw_line = FinPolyLine(self, [p0, p1], closed=False, pen=pg.mkPen(draw_line_color), movable=False)
            self.draw_line.setZValue(40)
            self.rois.append(self.draw_line)
            self.addItem(self.draw_line)
            self.drawing = True
        else:
            # draw placed point at end of poly-line
            self.draw_line.movePoint(-1, p1)
        if ev.isFinish():
            self.drawing = False
        ev.accept()

    def mouseMiddleDrag(self, ev, axis):
        '''Ctrl+MButton draw ellipses.'''
        if ev.modifiers() != QtCore.Qt.ControlModifier:
            return super().mouseDragEvent(ev, axis)
        p1 = self.mapToView(ev.pos())
        p1 = _clamp_point(self.parent(), p1)
        def nonzerosize(a, b):
            c = b-a
            return pg.Point(abs(c.x()) or 1, abs(c.y()) or 1e-3)
        if not self.drawing:
            # add new ellipse
            p0 = self.mapToView(ev.lastPos())
            p0 = _clamp_point(self.parent(), p0)
            s = nonzerosize(p0, p1)
            p0 = QtCore.QPointF(p0.x()-s.x()/2, p0.y()-s.y()/2)
            self.draw_ellipse = FinEllipse(p0, s, pen=pg.mkPen(draw_line_color), movable=True)
            self.draw_ellipse.setZValue(80)
            self.rois.append(self.draw_ellipse)
            self.addItem(self.draw_ellipse)
            self.drawing = True
        else:
            c = self.draw_ellipse.pos() + self.draw_ellipse.size()*0.5
            s = nonzerosize(c, p1)
            self.draw_ellipse.setSize(s*2, update=False)
            self.draw_ellipse.setPos(c-s)
        if ev.isFinish():
            self.drawing = False
        ev.accept()

    def mouseRightDrag(self, ev, axis):
        '''RButton is box zoom. At least for now.'''
        ev.accept()
        if not ev.isFinish():
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        else:
            self.rbScaleBox.hide()
            ax = QtCore.QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(ev.pos()))
            ax = self.childGroup.mapRectFromParent(ax)
            if ax.width() < 2: # zooming this narrow is probably a mistake
                ax.adjust(-1, 0, +1, 0)
            self.showAxRect(ax)
            self.axHistoryPointer += 1
            self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]

    def mouseClickEvent(self, ev):
        if self.master_viewbox:
            return self.master_viewbox.mouseClickEvent(ev)
        if _mouse_clicked(self, ev):
            ev.accept()
            return
        if ev.button() != QtCore.Qt.LeftButton or ev.modifiers() != QtCore.Qt.ControlModifier or not self.draw_line:
            return super().mouseClickEvent(ev)
        # add another segment to the currently drawn line
        p = self.mapToView(ev.pos())
        p = _clamp_point(self.parent(), p)
        self.append_draw_segment(p)
        self.drawing = False
        ev.accept()

    def keyPressEvent(self, ev):
        if self.master_viewbox:
            return self.master_viewbox.keyPressEvent(ev)
        if _key_pressed(self, ev):
            ev.accept()
            return
        super().keyPressEvent(ev)

    def linkedViewChanged(self, view, axis):
        if not self.datasrc or self.updating_linked:
            return
        if view and self.datasrc and view.datasrc:
            self.updating_linked = True
            tr = self.targetRect()
            vr = view.targetRect()
            is_dirty = view.force_range_update > 0
            is_same_scale = self.datasrc.xlen == view.datasrc.xlen
            if is_same_scale: # stable zoom based on index
                if is_dirty or abs(vr.left()-tr.left()) >= 1 or abs(vr.right()-tr.right()) >= 1:
                    if is_dirty:
                        view.force_range_update -= 1
                    self.update_y_zoom(vr.left(), vr.right())
            else: # sloppy one based on time stamps
                tt0,tt1,_,_,_ = self.datasrc.hilo(tr.left(), tr.right())
                vt0,vt1,_,_,_ = view.datasrc.hilo(vr.left(), vr.right())
                period2 = self.datasrc.period * 0.5 * 1e9
                if is_dirty or abs(vt0-tt0) >= period2 or abs(vt1-tt1) >= period2:
                    if is_dirty:
                        view.force_range_update -= 1
                    if self.parent():
                        x0,x1 = _pdtime2index(self.parent(), pd.Series([vt0,vt1]), any_end=True)
                        self.update_y_zoom(x0, x1)
            self.updating_linked = False

    def zoom_rect(self, vr, scale_fact, center):
        if not self.datasrc:
            return
        x0 = center.x() + (vr.left()-center.x()) * scale_fact
        x1 = center.x() + (vr.right()-center.x()) * scale_fact
        self.update_y_zoom(x0, x1)

    def pan_x(self, steps=None, percent=None):
        if self.datasrc is None:
            return
        if steps is None:
            steps = int(percent/100*self.targetRect().width())
        tr = self.targetRect()
        x1 = tr.right() + steps
        startx = -side_margin
        endx = self.datasrc.xlen - side_margin
        if x1 > endx:
            x1 = endx
        x0 = x1 - tr.width()
        if x0 < startx:
            x0 = startx
            x1 = x0 + tr.width()
        self.update_y_zoom(x0, x1)

    def refresh_all_y_zoom(self):
        '''This updates Y zoom on all views, such as when a mouse drag is completed.'''
        main_vb = self
        if self.linkedView(0):
            self.force_range_update = 1 # main need to update only once to us
            main_vb = list(self.win.axs)[0].vb
        main_vb.force_range_update = len(self.win.axs)-1 # update main as many times as there are other rows
        self.update_y_zoom()
        # refresh crosshair when done
        _mouse_moved(self.win, None)

    def update_y_zoom(self, x0=None, x1=None):
        datasrc = self.datasrc_or_standalone
        if datasrc is None:
            return
        if x0 is None or x1 is None:
            tr = self.targetRect()
            x0 = tr.left()
            x1 = tr.right()
        # make edges rigid
        xl = max(_round(x0-side_margin)+side_margin, -side_margin)
        xr = min(_round(x1-side_margin)+side_margin, datasrc.xlen-side_margin)
        dxl = xl-x0
        dxr = xr-x1
        if dxl > 0:
            x1 += dxl
        if dxr < 0:
            x0 += dxr
        x0 = max(_round(x0-side_margin)+side_margin, -side_margin)
        x1 = min(_round(x1-side_margin)+side_margin, datasrc.xlen-side_margin)
        # fetch hi-lo and set range
        _,_,hi,lo,cnt = datasrc.hilo(x0, x1)
        vr = self.viewRect()
        minlen = int((max_zoom_points-0.5) * self.max_zoom_points_f + 0.51)
        if (x1-x0) < vr.width() and cnt < minlen:
            return
        if not self.v_autozoom:
            hi = vr.bottom()
            lo = vr.top()
        if self.yscale.scaletype == 'log':
            lo = max(1e-100, lo)
            rng = (hi / lo) ** (1/self.v_zoom_scale)
            rng = min(rng, 1e50) # avoid float overflow
            base = (hi*lo) ** self.v_zoom_baseline
            y0 = base / rng**self.v_zoom_baseline
            y1 = base * rng**(1-self.v_zoom_baseline)
        else:
            rng = (hi-lo) / self.v_zoom_scale
            rng = max(rng, 2e-7) # some very weird bug where high/low exponents stops rendering
            base = (hi+lo) * self.v_zoom_baseline
            y0 = base - rng*self.v_zoom_baseline
            y1 = base + rng*(1-self.v_zoom_baseline)
        if not self.x_indexed:
            x0,x1 = _xminmax(datasrc, x_indexed=False, extra_margin=0)
        return self.set_range(x0, y0, x1, y1)

    def set_range(self, x0, y0, x1, y1):
        if x0 is None or x1 is None:
            tr = self.targetRect()
            x0 = tr.left()
            x1 = tr.right()
        if np.isnan(y0) or np.isnan(y1):
            return
        _y0 = self.yscale.invxform(y0, verify=True)
        _y1 = self.yscale.invxform(y1, verify=True)
        self.setRange(QtCore.QRectF(pg.Point(x0, _y0), pg.Point(x1, _y1)), padding=0)
        return True

    def remove_last_roi(self):
        if self.rois:
            if not isinstance(self.rois[-1], pg.PolyLineROI):
                self.removeItem(self.rois[-1])
                self.rois = self.rois[:-1]
            else:
                h = self.rois[-1].handles[-1]['item']
                self.rois[-1].removeHandle(h)
                if not self.rois[-1].segments:
                    self.removeItem(self.rois[-1])
                    self.rois = self.rois[:-1]
                    self.draw_line = None
            if self.rois:
                if isinstance(self.rois[-1], pg.PolyLineROI):
                    self.draw_line = self.rois[-1]
                    self.set_draw_line_color(draw_line_color)
            return True

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
    def __init__(self, ax, datasrc, lod):
        super().__init__()
        self.ax = ax
        self.datasrc = datasrc
        self.picture = QtGui.QPicture()
        self.painter = QtGui.QPainter()
        self.dirty = True
        self.lod = lod
        self.cachedRect = None

    def repaint(self):
        self.dirty = True
        self.paint(self.painter)

    def paint(self, p, *args):
        if self.datasrc.is_sparse:
            self.dirty = True
        self.update_dirty_picture(self.viewRect())
        p.drawPicture(0, 0, self.picture)

    def update_dirty_picture(self, visibleRect):
        if self.dirty or \
            (self.lod and # regenerate when zoom changes?
                (visibleRect.left() < self.cachedRect.left() or \
                 visibleRect.right() > self.cachedRect.right() or \
                 visibleRect.width() < self.cachedRect.width() / cache_candle_factor)): # optimize when zooming in
            self._generate_picture(visibleRect)

    def _generate_picture(self, boundingRect):
        w = boundingRect.width()
        self.cachedRect = QtCore.QRectF(boundingRect.left()-(cache_candle_factor-1)*0.5*w, 0, cache_candle_factor*w, 0)
        self.painter.begin(self.picture)
        self._generate_dummy_picture(self.viewRect())
        self.generate_picture(self.cachedRect)
        self.painter.end()
        self.dirty = False

    def _generate_dummy_picture(self, boundingRect):
        if self.datasrc.is_sparse:
            # just draw something to ensure PyQt will paint us again
            self.painter.setPen(pg.mkPen(background))
            self.painter.setBrush(pg.mkBrush(background))
            l,r = boundingRect.left(), boundingRect.right()
            self.painter.drawRect(QtCore.QRectF(l, boundingRect.top(), 1e-3, boundingRect.height()*1e-5))
            self.painter.drawRect(QtCore.QRectF(r, boundingRect.bottom(), -1e-3, -boundingRect.height()*1e-5))

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())



class CandlestickItem(FinPlotItem):
    def __init__(self, ax, datasrc, draw_body, draw_shadow, candle_width, colorfunc):
        self.colors = dict(bull_shadow      = candle_bull_color,
                           bull_frame       = candle_bull_color,
                           bull_body        = candle_bull_body_color,
                           bear_shadow      = candle_bear_color,
                           bear_frame       = candle_bear_color,
                           bear_body        = candle_bear_color,
                           weak_bull_shadow = brighten(candle_bull_color, 1.2),
                           weak_bull_frame  = brighten(candle_bull_color, 1.2),
                           weak_bull_body   = brighten(candle_bull_color, 1.2),
                           weak_bear_shadow = brighten(candle_bear_color, 1.5),
                           weak_bear_frame  = brighten(candle_bear_color, 1.5),
                           weak_bear_body   = brighten(candle_bear_color, 1.5))
        self.draw_body = draw_body
        self.draw_shadow = draw_shadow
        self.candle_width = candle_width
        self.shadow_width = candle_shadow_width
        self.colorfunc = colorfunc
        self.x_offset = 0
        super().__init__(ax, datasrc, lod=True)

    def generate_picture(self, boundingRect):
        w = self.candle_width
        w2 = w * 0.5
        left,right = boundingRect.left(), boundingRect.right()
        p = self.painter
        df,origlen = self.datasrc.rows(5, left, right, yscale=self.ax.vb.yscale)
        drawing_many_shadows = self.draw_shadow and origlen > lod_candles*2//3
        for shadow,frame,body,df_rows in self.colorfunc(self, self.datasrc, df):
            idxs = df_rows.index
            rows = df_rows.values
            if self.x_offset:
                idxs += self.x_offset
            if self.draw_shadow:
                p.setPen(pg.mkPen(shadow, width=self.shadow_width))
                for x,(t,open,close,high,low) in zip(idxs, rows):
                    if high > low:
                        p.drawLine(QtCore.QPointF(x, low), QtCore.QPointF(x, high))
            if self.draw_body and not drawing_many_shadows: # settle with only drawing shadows if too much detail
                p.setPen(pg.mkPen(frame))
                p.setBrush(pg.mkBrush(body))
                for x,(t,open,close,high,low) in zip(idxs, rows):
                    p.drawRect(QtCore.QRectF(x-w2, open, w, close-open))

    def rowcolors(self, prefix):
        return [self.colors[prefix+'_shadow'], self.colors[prefix+'_frame'], self.colors[prefix+'_body']]



class HeatmapItem(FinPlotItem):
    def __init__(self, ax, datasrc, rect_size=0.9, filter_limit=0, colmap=colmap_clash, whiteout=0.0, colcurve=lambda x:pow(x,4)):
        self.rect_size = rect_size
        self.filter_limit = filter_limit
        self.colmap = colmap
        self.whiteout = whiteout
        self.colcurve = colcurve
        self.col_data_end = len(datasrc.df.columns)
        super().__init__(ax, datasrc, lod=False)

    def generate_picture(self, boundingRect):
        prices = self.datasrc.df.columns[self.datasrc.col_data_offset:self.col_data_end]
        h0 = (prices[0] - prices[1]) * (1-self.rect_size)
        h1 = (prices[0] - prices[1]) * (1-(1-self.rect_size)*2)
        rect_size2 = 0.5 * self.rect_size
        df = self.datasrc.df.iloc[:, self.datasrc.col_data_offset:self.col_data_end]
        values = df.values
        # normalize
        values -= np.nanmin(values)
        values /= np.nanmax(values) / (1+self.whiteout) # overshoot for coloring
        lim = self.filter_limit * (1+self.whiteout)
        p = self.painter
        for t,row in enumerate(values):
            for ci,price in enumerate(prices):
                v = row[ci]
                if v >= lim:
                    v = 1 - self.colcurve(1 - (v-lim)/(1-lim))
                    color = self.colmap.map(v, mode='qcolor')
                    p.fillRect(QtCore.QRectF(t-rect_size2, price+h0, self.rect_size, h1), color)



class HorizontalTimeVolumeItem(CandlestickItem):
    def __init__(self, ax, datasrc, candle_width=0.8, draw_va=0.0, draw_body=0.4, draw_poc=0.0, colorfunc=None):
        '''A negative draw_body does not mean that the candle is drawn in the opposite direction (use negative volume for that),
           but instead that screen scale will be used instead of interval-relative scale.'''
        self.draw_va = draw_va
        self.draw_poc = draw_poc
        ## self.col_data_end = len(datasrc.df.columns)
        colorfunc = colorfunc or horizvol_colorfilter() # resolve function lower down in source code
        super().__init__(ax, datasrc, draw_shadow=False, candle_width=candle_width, draw_body=draw_body, colorfunc=colorfunc)
        self.lod = False
        self.colors.update(dict(neutral_shadow  = volume_neutral_color,
                                neutral_frame   = volume_neutral_color,
                                neutral_body    = volume_neutral_color,
                                bull_body       = candle_bull_color))

    def generate_picture(self, boundingRect):
        times = self.datasrc.df.iloc[:, 0]
        vals = self.datasrc.df.values
        prices = vals[:, self.datasrc.col_data_offset::2]
        volumes = vals[:, self.datasrc.col_data_offset+1::2].T
        # normalize
        try:
            f = self.datasrc.period / _get_datasrc(self.ax).period
            times = _pdtime2index(self.ax, times, require_time=True)
        except AssertionError:
            f = 1
        draw_body = self.draw_body
        if draw_body < 0:
            f *= -draw_body * self.ax.vb.targetRect().width()
            draw_body = 1
        binc = len(volumes)
        if not binc:
            return
        divvol = np.nanmax(np.abs(volumes), axis=0)
        divvol[divvol==0] = 1
        volumes = (volumes * f / divvol).T
        p = self.painter
        h = 1e-10
        for i in self.datasrc.df.index:
            prcr = prices[i]
            prv = prcr[~np.isnan(prcr)]
            if len(prv) > 1:
                h = np.diff(prv).min()
            t = times[i]
            volr = np.nan_to_num(volumes[i])

            # calc poc
            pocidx = np.nanargmax(volr)

            # draw value area
            if self.draw_va:
                volrs = volr / np.nansum(volr)
                v = volrs[pocidx]
                a = b = pocidx
                while a>=0 or b<binc:
                    if v >= self.draw_va:
                        break
                    aa = a - 1
                    bb = b + 1
                    va = volrs[aa] if aa>=0 else 0
                    vb = volrs[bb] if bb<binc else 0
                    if va >= vb: # NOTE both == is also ok
                        a = max(0, aa)
                        v += va
                    if va <= vb: # NOTE both == is also ok
                        b = min(binc-1, bb)
                        v += vb
                color = pg.mkColor(band_color)
                p.fillRect(QtCore.QRectF(t, prcr[a], f, prcr[b]-prcr[a]+h), color)

            # draw horizontal bars
            if draw_body:
                h0 = h * (1-self.candle_width)/2
                h1 = h * self.candle_width
                for shadow,frame,body,data in self.colorfunc(self, self.datasrc, np.array([prcr, volr])):
                    p.setPen(pg.mkPen(frame))
                    p.setBrush(pg.mkBrush(body))
                    prcr_,volr_ = data
                    for w,y in zip(volr_, prcr_):
                        if abs(w) > 1e-15:
                            p.drawRect(QtCore.QRectF(t, y+h0, w*draw_body, h1))

            # draw poc line
            if self.draw_poc:
                y = prcr[pocidx] + h / 2
                p.setPen(pg.mkPen(poc_color))
                p.drawLine(QtCore.QPointF(t, y), QtCore.QPointF(t+f*self.draw_poc, y))



class ScatterLabelItem(FinPlotItem):
    def __init__(self, ax, datasrc, color, anchor):
        self.color = color
        self.text_items = {}
        self.anchor = anchor
        self.show = False
        super().__init__(ax, datasrc, lod=True)

    def generate_picture(self, bounding_rect):
        rows = self.getrows(bounding_rect)
        if len(rows) > lod_labels: # don't even generate when there's too many of them
            self.clear_items(list(self.text_items.keys()))
            return
        drops = set(self.text_items.keys())
        created = 0
        for x,t,y,txt in rows:
            txt = str(txt)
            ishtml = '<' in txt and '>' in txt
            key = '%s:%.8f' % (t, y)
            if key in self.text_items:
                item = self.text_items[key]
                (item.setHtml if ishtml else item.setText)(txt)
                item.setPos(x, y)
                drops.remove(key)
            else:
                kws = {'html':txt} if ishtml else {'text':txt}
                self.text_items[key] = item = pg.TextItem(color=self.color, anchor=self.anchor, **kws)
                item.setPos(x, y)
                item.setParentItem(self)
                created += 1
        if created > 0 or self.dirty: # only reduce cache if we've added some new or updated
            self.clear_items(drops)

    def clear_items(self, drop_keys):
        for key in drop_keys:
            item = self.text_items[key]
            item.scene().removeItem(item)
            del self.text_items[key]

    def getrows(self, bounding_rect):
        left,right = bounding_rect.left(), bounding_rect.right()
        df,_ = self.datasrc.rows(3, left, right, yscale=self.ax.vb.yscale, lod=False)
        rows = df.dropna()
        idxs = rows.index
        rows = rows.values
        rows = [(i,t,y,txt) for i,(t,y,txt) in zip(idxs, rows) if txt]
        return rows

    def boundingRect(self):
        return self.viewRect()


def create_plot(title='Finance Plot', rows=1, init_zoom_periods=1e10, maximize=True, yscale='linear'):
    pg.setConfigOptions(foreground=foreground, background=background)
    win = FinWindow(title)
    # normally first graph is of higher significance, so enlarge
    win.ci.layout.setRowStretchFactor(0, top_graph_scale)
    win.show_maximized = maximize
    ax0 = axs = create_plot_widget(master=win, rows=rows, init_zoom_periods=init_zoom_periods, yscale=yscale)
    axs = axs if type(axs) in (tuple,list) else [axs]
    for ax in axs:
        win.addItem(ax)
        win.nextRow()
    return ax0


def create_plot_widget(master, rows=1, init_zoom_periods=1e10, yscale='linear'):
    pg.setConfigOptions(foreground=foreground, background=background)
    global last_ax
    if master not in windows:
        windows.append(master)
    axs = []
    prev_ax = None
    for n in range(rows):
        ysc = yscale[n] if type(yscale) in (list,tuple) else yscale
        ysc = YScale(ysc, 1)
        viewbox = FinViewBox(master, init_steps=init_zoom_periods, yscale=ysc, v_zoom_scale=1-y_pad, enableMenu=False)
        ax = prev_ax = _add_timestamp_plot(master=master, prev_ax=prev_ax, viewbox=viewbox, index=n, yscale=ysc)
        if axs:
            ax.setXLink(axs[0].vb)
        else:
            viewbox.setFocus()
        axs += [ax]
    if isinstance(master, pg.GraphicsLayoutWidget):
        proxy = pg.SignalProxy(master.scene().sigMouseMoved, rateLimit=144, slot=partial(_mouse_moved, master))
    else:
        proxy = []
        for ax in axs:
            proxy += [pg.SignalProxy(ax.ax_widget.scene().sigMouseMoved, rateLimit=144, slot=partial(_mouse_moved, master))]
    master_data[master] = dict(proxymm=proxy, last_mouse_evs=None, last_mouse_y=0)
    last_ax = axs[0]
    return axs[0] if len(axs) == 1 else axs


def close():
    for win in windows:
        try:
            win.close()
        except Exception as e:
            print('Window closing error:', type(e), e)
    global last_ax
    windows.clear()
    overlay_axs.clear()
    _clear_timers()
    sounds.clear()
    master_data.clear()
    last_ax = None


def price_colorfilter(item, datasrc, df):
    opencol = df.columns[1]
    closecol = df.columns[2]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    yield item.rowcolors('bull') + [df.loc[is_up, :]]
    yield item.rowcolors('bear') + [df.loc[~is_up, :]]


def volume_colorfilter(item, datasrc, df):
    opencol = df.columns[3]
    closecol = df.columns[4]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    yield item.rowcolors('bull') + [df.loc[is_up, :]]
    yield item.rowcolors('bear') + [df.loc[~is_up, :]]


def strength_colorfilter(item, datasrc, df):
    opencol = df.columns[1]
    closecol = df.columns[2]
    startcol = df.columns[3]
    endcol = df.columns[4]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    is_strong = df[startcol] <= df[endcol]
    yield item.rowcolors('bull') + [df.loc[is_up&is_strong, :]]
    yield item.rowcolors('weak_bull') + [df.loc[is_up&(~is_strong), :]]
    yield item.rowcolors('weak_bear') + [df.loc[(~is_up)&is_strong, :]]
    yield item.rowcolors('bear') + [df.loc[(~is_up)&(~is_strong), :]]


def volume_colorfilter_section(sections=[]):
    '''The sections argument is a (starting_index, color_name) array.'''
    def _colorfilter(sections, item, datasrc, df):
        if not sections:
            return volume_colorfilter(item, datasrc, df)
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = df.iloc[i0:i1, :]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


def horizvol_colorfilter(sections=[]):
    '''The sections argument is a (starting_index, color_name) array.'''
    def _colorfilter(sections, item, datasrc, data):
        if not sections:
            yield item.rowcolors('neutral') + [data]
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = data[:, i0:i1]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


def candlestick_ochl(datasrc, draw_body=True, draw_shadow=True, candle_width=0.6, ax=None, colorfunc=price_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, datasrc)
    datasrc.scale_cols = [3,4] # only hi+lo scales
    _set_datasrc(ax, datasrc)
    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=draw_body, draw_shadow=draw_shadow, candle_width=candle_width, colorfunc=colorfunc)
    _update_significants(ax, datasrc, force=True)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.addItem(item)
    return item


def renko(x, y=None, bins=None, step=None, ax=None, colorfunc=price_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, x, y)
    origdf = datasrc.df
    adj = _adjust_renko_log_datasrc if ax.vb.yscale.scaletype == 'log' else _adjust_renko_datasrc
    step_adjust_renko_datasrc = partial(adj, bins, step)
    step_adjust_renko_datasrc(datasrc)
    ax.decouple()
    item = candlestick_ochl(datasrc, draw_shadow=False, candle_width=1, ax=ax, colorfunc=colorfunc)
    item.colors['bull_body'] = item.colors['bull_frame']
    item.update_data = partial(_update_data, None, step_adjust_renko_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    global epoch_period
    epoch_period = (origdf.iloc[1,0] - origdf.iloc[0,0]) // int(1e9)
    return item


def volume_ocv(datasrc, candle_width=0.8, ax=None, colorfunc=volume_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, datasrc)
    _adjust_volume_datasrc(datasrc)
    _set_datasrc(ax, datasrc)
    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=True, draw_shadow=False, candle_width=candle_width, colorfunc=colorfunc)
    _update_significants(ax, datasrc, force=True)
    item.colors['bull_body'] = item.colors['bull_frame']
    if colorfunc == volume_colorfilter: # assume normal volume plot
        item.colors['bull_frame'] = volume_bull_color
        item.colors['bull_body']  = volume_bull_body_color
        item.colors['bear_frame'] = volume_bear_color
        item.colors['bear_body']  = volume_bear_color
        ax.vb.v_zoom_baseline = 0
    else:
        item.colors['weak_bull_frame'] = brighten(volume_bull_color, 1.2)
        item.colors['weak_bull_body']  = brighten(volume_bull_color, 1.2)
    item.update_data = partial(_update_data, None, _adjust_volume_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.addItem(item)
    item.setZValue(-20)
    return item


def horiz_time_volume(datasrc, ax=None, **kwargs):
    '''Draws multiple fixed horizontal volumes. The input format is:
       [[time0, [(price0,volume0),(price1,volume1),...]], ...]

       This chart needs to be plot last, so it knows if it controls
       what time periods are shown, or if its using time already in
       place by another plot.'''
    # update handling default if necessary
    global max_zoom_points, right_margin_candles
    if max_zoom_points > 15:
        max_zoom_points = 4
    if right_margin_candles > 3:
        right_margin_candles = 1

    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _preadjust_horiz_datasrc(datasrc)
    datasrc = _create_datasrc(ax, datasrc)
    _adjust_horiz_datasrc(datasrc)
    if ax.vb.datasrc is not None:
        datasrc.standalone = True # only force standalone if there is something on our charts already
    datasrc.scale_cols = [datasrc.col_data_offset, len(datasrc.df.columns)-2] # first and last price columns
    datasrc.pre_update = lambda df: df.loc[:, :df.columns[0]] # throw away previous data
    datasrc.post_update = lambda df: df.dropna(how='all') # kill all-NaNs
    _set_datasrc(ax, datasrc)
    item = HorizontalTimeVolumeItem(ax=ax, datasrc=datasrc, **kwargs)
    item.update_data = partial(_update_data, _preadjust_horiz_datasrc, _adjust_horiz_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    item.setZValue(-10)
    ax.addItem(item)
    return item


def heatmap(datasrc, ax=None, **kwargs):
    '''Expensive function. Only use on small data sets. See HeatmapItem for kwargs. Input datasrc
       has x (time) in index or first column, y (price) as column names, and intensity (color) as
       cell values.'''
    ax = _create_plot(ax=ax, maximize=False)
    if ax.vb.v_zoom_scale >= 0.9:
        ax.vb.v_zoom_scale = 0.6
    datasrc = _create_datasrc(ax, datasrc)
    datasrc.scale_cols = [] # doesn't scale
    _set_datasrc(ax, datasrc)
    item = HeatmapItem(ax=ax, datasrc=datasrc, **kwargs)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    item.setZValue(-30)
    ax.addItem(item)
    if ax.vb.datasrc is not None and not ax.vb.datasrc.timebased(): # manual zoom update
        ax.setXLink(None)
        if ax.prev_ax:
            ax.prev_ax.set_visible(xaxis=True)
        df = ax.vb.datasrc.df
        prices = df.columns[ax.vb.datasrc.col_data_offset:item.col_data_end]
        delta_price = abs(prices[0] - prices[1])
        ax.vb.set_range(0, min(df.columns[1:]), len(df), max(df.columns[1:])+delta_price)
    return item


def bar(x, y=None, width=0.8, ax=None, colorfunc=strength_colorfilter, **kwargs):
    '''Bar plots are decoupled. Use volume_ocv() if you want a bar plot which relates to other time plots.'''
    global right_margin_candles, max_zoom_points
    right_margin_candles = 0
    max_zoom_points = min(max_zoom_points, 8)
    ax = _create_plot(ax=ax, maximize=False)
    ax.decouple()
    datasrc = _create_datasrc(ax, x, y)
    _adjust_bar_datasrc(datasrc, order_cols=False) # don't rearrange columns, done for us in volume_ocv()
    item = volume_ocv(datasrc, candle_width=width, ax=ax, colorfunc=colorfunc)
    item.update_data = partial(_update_data, None, _adjust_bar_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.vb.pre_process_data()
    if ax.vb.y_min >= 0:
        ax.vb.v_zoom_baseline = 0
    return item


def hist(x, bins, ax=None, **kwargs):
    hist_data = pd.cut(x, bins=bins).value_counts()
    data = [(i.mid,0,hist_data.loc[i],hist_data.loc[i]) for i in sorted(hist_data.index)]
    df = pd.DataFrame(data, columns=['x','_op_','_cl_','bin'])
    df.set_index('x', inplace=True)
    item = bar(df, ax=ax)
    del item.update_data
    return item


def plot(x, y=None, color=None, width=1, ax=None, style=None, legend=None, zoomscale=True, **kwargs):
    ax = _create_plot(ax=ax, maximize=False)
    used_color = _get_color(ax, style, color)
    datasrc = _create_datasrc(ax, x, y)
    if not zoomscale:
        datasrc.scale_cols = []
    _set_datasrc(ax, datasrc)
    if legend is not None:
        _create_legend(ax)
    x = datasrc.x if datasrc.standalone or not ax.vb.x_indexed else datasrc.index
    y = datasrc.y / ax.vb.yscale.scalef
    if style is None or any(ch in style for ch in '-_.'):
        connect_dots = 'finite' # same as matplotlib; use datasrc.standalone=True if you want to keep separate intervals on a plot
        item = ax.plot(x, y, pen=_makepen(color=used_color, style=style, width=width), name=legend, connect=connect_dots)
        item.setZValue(5)
    else:
        symbol = {'v':'t', '^':'t1', '>':'t2', '<':'t3'}.get(style, style) # translate some similar styles
        yfilter = y.notnull()
        ser = y.loc[yfilter]
        x = x.loc[yfilter].values if hasattr(x, 'loc') else x[yfilter]
        item = ax.plot(x, ser.values, pen=None, symbol=symbol, symbolPen=None, symbolSize=7*width, symbolBrush=pg.mkBrush(used_color), name=legend)
        if width < 1:
            item.opts['antialias'] = True
        item.scatter._dopaint = item.scatter.paint
        item.scatter.paint = partial(_paint_scatter, item.scatter)
        # optimize (when having large number of points) by ignoring scatter click detection
        _dummy_mouse_click = lambda ev: 0
        item.scatter.mouseClickEvent = _dummy_mouse_click
        item.setZValue(10)
    item.opts['handed_color'] = color
    item.ax = ax
    item.datasrc = datasrc
    _update_significants(ax, datasrc, force=False)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    # add legend to main ax, not to overlay
    axm = ax.vb.master_viewbox.parent() if ax.vb.master_viewbox else ax
    if axm.legend is not None:
        if legend and axm != ax:
            axm.legend.addItem(item, name=legend)
        for _,label in axm.legend.items:
            if label.text == legend:
                label.setAttr('justify', 'left')
                label.setText(label.text, color=legend_text_color)
    return item


def labels(x, y=None, labels=None, color=None, ax=None, anchor=(0.5,1)):
    ax = _create_plot(ax=ax, maximize=False)
    used_color = _get_color(ax, '?', color)
    datasrc = _create_datasrc(ax, x, y, labels)
    datasrc.scale_cols = [] # don't use this for scaling
    _set_datasrc(ax, datasrc)
    item = ScatterLabelItem(ax=ax, datasrc=datasrc, color=used_color, anchor=anchor)
    _update_significants(ax, datasrc, force=False)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.addItem(item)
    if ax.vb.v_zoom_scale > 0.9: # adjust to make hi/lo text fit
        ax.vb.v_zoom_scale = 0.9
    return item


def add_legend(text, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    _create_legend(ax)
    row = ax.legend.layout.rowCount()
    label = pg.LabelItem(text, color=legend_text_color, justify='left')
    ax.legend.layout.addItem(label, row, 0, 1, 2)
    return label


def fill_between(plot0, plot1, color=None):
    used_color = brighten(_get_color(plot0.ax, None, color), 1.3)
    item = pg.FillBetweenItem(plot0, plot1, brush=pg.mkBrush(used_color))
    item.ax = plot0.ax
    item.setZValue(-40)
    item.ax.addItem(item)
    return item


def set_x_pos(xmin, xmax, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    xidx0,xidx1 = _pdtime2index(ax, pd.Series([xmin, xmax]))
    ax.vb.update_y_zoom(xidx0, xidx1)
    _repaint_candles()


def set_y_range(ymin, ymax, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    ax.setLimits(yMin=ymin, yMax=ymax)
    ax.vb.v_autozoom = False
    ax.vb.set_range(None, ymin, None, ymax)


def set_y_scale(yscale='linear', ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    ax.setLogMode(y=(yscale=='log'))
    ax.vb.yscale = YScale(yscale, ax.vb.yscale.scalef)


def add_band(y0, y1, color=band_color, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    color = _get_color(ax, None, color)
    ix = ax.vb.yscale.invxform
    lr = pg.LinearRegionItem([ix(y0),ix(y1)], orientation=pg.LinearRegionItem.Horizontal, brush=pg.mkBrush(color), movable=False)
    lr.lines[0].setPen(pg.mkPen(None))
    lr.lines[1].setPen(pg.mkPen(None))
    lr.setZValue(-50)
    lr.ax = ax
    ax.addItem(lr)
    return lr


def add_rect(p0, p1, color=band_color, interactive=False, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    x_pts = _pdtime2index(ax, pd.Series([p0[0], p1[0]]))
    ix = ax.vb.yscale.invxform
    y0,y1 = sorted([p0[1], p1[1]])
    pos  = (x_pts[0], ix(y0))
    size = (x_pts[1]-pos[0], ix(y1-y0))
    rect = FinRect(ax=ax, brush=pg.mkBrush(color), pos=pos, size=size, movable=interactive, resizable=interactive, rotatable=False)
    rect.setZValue(-40)
    if interactive:
        ax.vb.rois.append(rect)
    rect.ax = ax
    ax.addItem(rect)
    return rect


def add_line(p0, p1, color=draw_line_color, width=1, style=None, interactive=False, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    used_color = _get_color(ax, style, color)
    pen = _makepen(color=used_color, style=style, width=width)
    x_pts = _pdtime2index(ax, pd.Series([p0[0], p1[0]]))
    ix = ax.vb.yscale.invxform
    pts = [(x_pts[0], ix(p0[1])), (x_pts[1], ix(p1[1]))]
    if interactive:
        line = FinPolyLine(ax.vb, pts, closed=False, pen=pen, movable=False)
        ax.vb.rois.append(line)
    else:
        line = FinLine(pts, pen=pen)
    line.ax = ax
    ax.addItem(line)
    return line


def add_text(pos, s, color=draw_line_color, anchor=(0,0), ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    color = _get_color(ax, None, color)
    text = pg.TextItem(s, color=color, anchor=anchor)
    x = pos[0]
    if ax.vb.datasrc is not None:
        x = _pdtime2index(ax, pd.Series([pos[0]]))[0]
    y = ax.vb.yscale.invxform(pos[1])
    text.setPos(x, y)
    text.setZValue(50)
    text.ax = ax
    ax.addItem(text, ignoreBounds=True)
    return text


def remove_line(line):
    print('remove_line() is deprecated, use remove_primitive() instead')
    remove_primitive(line)


def remove_text(text):
    print('remove_text() is deprecated, use remove_primitive() instead')
    remove_primitive(text)


def remove_primitive(primitive):
    ax = primitive.ax
    ax.removeItem(primitive)
    if primitive in ax.vb.rois:
        ax.vb.rois.remove(primitive)
    if hasattr(primitive, 'texts'):
        for txt in primitive.texts:
            ax.vb.removeItem(txt)


def set_time_inspector(inspector, ax=None, when='click'):
    '''Callback when clicked like so: inspector(x, y).'''
    ax = ax if ax else last_ax
    win = ax.vb.win
    if when == 'hover':
        win.proxy_hover = pg.SignalProxy(win.scene().sigMouseMoved, rateLimit=15, slot=partial(_inspect_pos, ax, inspector))
    elif when in ('dclick', 'double-click'):
        win.proxy_dclick = pg.SignalProxy(win.scene().sigMouseClicked, slot=partial(_inspect_clicked, ax, inspector, True))
    else:
        win.proxy_click = pg.SignalProxy(win.scene().sigMouseClicked, slot=partial(_inspect_clicked, ax, inspector, False))


def add_crosshair_info(infofunc, ax=None):
    '''Callback when crosshair updated like so: info(ax,x,y,xtext,ytext); the info()
       callback must return two values: xtext and ytext.'''
    ax = _create_plot(ax=ax, maximize=False)
    ax.crosshair.infos.append(infofunc)


def timer_callback(update_func, seconds, single_shot=False):
    global timers
    timer = QtCore.QTimer()
    timer.timeout.connect(update_func)
    if single_shot:
        timer.setSingleShot(True)
    timer.start(int(seconds*1000))
    timers.append(timer)
    return timer


def autoviewrestore(enable=True):
    '''Restor functionality saves view zoom coordinates when closing a window, and
       load them when creating the plot (with the same name) again.'''
    global viewrestore
    viewrestore = enable


def refresh():
    for win in windows:
        vbs = [ax.vb for ax in win.axs] + [ax.vb for ax in overlay_axs if ax.vb.win==win]
        for vb in vbs:
            vb.pre_process_data()
        if viewrestore:
            if _loadwindata(win):
                continue
        _set_max_zoom(vbs)
        for vb in vbs:
            datasrc = vb.datasrc_or_standalone
            if datasrc and (vb.linkedView(0) is None or vb.linkedView(0).datasrc is None or vb.master_viewbox):
                vb.update_y_zoom(datasrc.init_x0, datasrc.init_x1)
    _repaint_candles()
    for win in windows:
        _mouse_moved(win, None)


def show(qt_exec=True):
    refresh()
    for win in windows:
        if isinstance(win, FinWindow) or qt_exec:
            if win.show_maximized:
                win.showMaximized()
            else:
                win.show()
    if windows and qt_exec:
        global last_ax, app
        app = QtGui.QApplication.instance()
        app.exec_()
        windows.clear()
        overlay_axs.clear()
        _clear_timers()
        sounds.clear()
        master_data.clear()
        last_ax = None


def play_sound(filename):
    if filename not in sounds:
        from PyQt5.QtMultimedia import QSound
        sounds[filename] = QSound(filename) # disallow gc
    s = sounds[filename]
    s.play()


def screenshot(file, fmt='png'):
    if _internal_windows_only() and not app:
        print('ERROR: screenshot must be callbacked from e.g. timer_callback()')
        return False
    try:
        buffer = QtCore.QBuffer()
        app.primaryScreen().grabWindow(windows[0].winId()).save(buffer, fmt)
        file.write(buffer.data())
        return True
    except Exception as e:
        print('Screenshot error:', type(e), e)
    return False


#################### INTERNALS ####################


def _openfile(*args):
    return open(*args)


def _loadwindata(win):
    try: os.mkdir(os.path.expanduser('~/.finplot'))
    except: pass
    try:
        f = os.path.expanduser('~/.finplot/'+win.title.replace('/','-')+'.ini')
        settings = [(k.strip(),literal_eval(v.strip())) for line in _openfile(f) for k,d,v in [line.partition('=')] if v]
    except:
        return
    kvs = {k:v for k,v in settings}
    vbs = set(ax.vb for ax in win.axs)
    zoom_set = False
    for vb in vbs:
        ds = vb.datasrc
        if ds and (vb.linkedView(0) is None or vb.linkedView(0).datasrc is None or vb.master_viewbox):
            period = ds.period
            if kvs['min_x'] >= ds.x.iloc[0]-period and kvs['max_x'] <= ds.x.iloc[-1]+period:
                x0,x1 = ds.x.loc[ds.x>=kvs['min_x']].index[0], ds.x.loc[ds.x<=kvs['max_x']].index[-1]
                if x1 == len(ds.x)-1:
                    x1 += right_margin_candles
                zoom_set = vb.update_y_zoom(x0, x1)
    return zoom_set


def _savewindata(win):
    if not viewrestore:
        return
    try:
        min_x = int(1e100)
        max_x = int(-1e100)
        for ax in win.axs:
            if ax.vb.targetRect().right() < 4: # ignore empty plots
                continue
            if ax.vb.datasrc is None:
                continue
            t0,t1,_,_,_ = ax.vb.datasrc.hilo(ax.vb.targetRect().left(), ax.vb.targetRect().right())
            min_x = np.nanmin([min_x, t0])
            max_x = np.nanmax([max_x, t1])
        if np.max(np.abs([min_x, max_x])) < 1e99:
            s = 'min_x = %s\nmax_x = %s\n' % (min_x, max_x)
            f = os.path.expanduser('~/.finplot/'+win.title.replace('/','-')+'.ini')
            try: changed = _openfile(f).read() != s
            except: changed = True
            if changed:
                _openfile(f, 'wt').write(s)
                ## print('%s saved' % win.title)
    except Exception as e:
        print('Error saving plot:', e)


def _internal_windows_only():
     return all(isinstance(win,FinWindow) for win in windows)


def _create_plot(ax=None, **kwargs):
    if ax:
        return ax
    if last_ax:
        return last_ax
    return create_plot(**kwargs)


def _clear_timers():
    for timer in timers:
        timer.timeout.disconnect()
    timers.clear()


def _add_timestamp_plot(master, prev_ax, viewbox, index, yscale):
    native_win = isinstance(master, pg.GraphicsLayoutWidget)
    if native_win and prev_ax is not None:
        prev_ax.set_visible(xaxis=False) # hide the whole previous axis
    axes = {'bottom': EpochAxisItem(vb=viewbox, orientation='bottom'),
            'left':   YAxisItem(vb=viewbox, orientation='left')}
    if native_win:
        ax = pg.PlotItem(viewBox=viewbox, axisItems=axes, name='plot-%i'%index, enableMenu=False)
    else:
        axw = pg.PlotWidget(viewBox=viewbox, axisItems=axes, name='plot-%i'%index, enableMenu=False)
        ax = axw.plotItem
        ax.ax_widget = axw
    ax.axes['left']['item'].setWidth(y_label_width) # this is to put all graphs on equal footing when texts vary from 0.4 to 2000000
    ax.axes['left']['item'].setStyle(tickLength=-5) # some bug, totally unexplicable (why setting the default value again would fix repaint width as axis scale down)
    ax.axes['left']['item'].setZValue(30) # put axis in front instead of behind data
    ax.axes['bottom']['item'].setZValue(30)
    ax.setLogMode(y=(yscale.scaletype=='log'))
    ax.significant_decimals = significant_decimals
    ax.significant_eps = significant_eps
    ax.crosshair = FinCrossHair(ax, color=cross_hair_color)
    ax.hideButtons()
    ax.overlay = partial(_ax_overlay, ax)
    ax.set_visible = partial(_ax_set_visible, ax)
    ax.decouple = partial(_ax_decouple, ax)
    ax.disable_x_index = partial(_ax_disable_x_index, ax)
    ax.reset = partial(_ax_reset, ax)
    ax.prev_ax = prev_ax
    ax.win_index = index
    if index%2:
        viewbox.setBackgroundColor(odd_plot_background)
    viewbox.setParent(ax)
    return ax


def _ax_overlay(ax, scale=0.25, yaxis=False):
    '''The scale parameter defines how "high up" on the initial plot this overlay will show.
       The yaxis parameter can be one of [False, 'linear', 'log'].'''
    yscale = yaxis if yaxis else 'linear'
    viewbox = FinViewBox(ax.vb.win, init_steps=ax.vb.init_steps, yscale=YScale(yscale, 1), enableMenu=False)
    viewbox.master_viewbox = ax.vb
    viewbox.setZValue(-5)
    viewbox.setBackgroundColor(ax.vb.state['background'])
    ax.vb.setBackgroundColor(None)
    viewbox.v_zoom_scale = scale
    if hasattr(ax, 'ax_widget'):
        ax.ax_widget.scene().addItem(viewbox)
    else:
        ax.vb.win.centralWidget.scene().addItem(viewbox)
    viewbox.setXLink(ax.vb)
    def updateView():
        viewbox.setGeometry(ax.vb.sceneBoundingRect())
    axo = pg.PlotItem(enableMenu=False)
    axo.significant_decimals = significant_decimals
    axo.significant_eps = significant_eps
    axo.vb = viewbox
    axo.prev_ax = None
    axo.crosshair = None
    axo.decouple = partial(_ax_decouple, axo)
    axo.disable_x_index = partial(_ax_disable_x_index, axo)
    axo.reset = partial(_ax_reset, axo)
    axo.hideAxis('left')
    axo.hideAxis('right')
    axo.hideAxis('bottom')
    axo.hideButtons()
    viewbox.addItem(axo)
    if yaxis and isinstance(axo.vb.win, pg.GraphicsLayoutWidget):
        axi = YAxisItem(vb=axo.vb, orientation='right')
        axo.axes['right'] = {'item':axi}
        axi.linkToView(axo.vb)
        row = ax.win_index
        for col in range(1, 100):
            if axo.vb.win.getItem(row, col) is None:
                axo.vb.win.addItem(axi, row=row, col=1)
                break
    ax.vb.sigResized.connect(updateView)
    overlay_axs.append(axo)
    updateView()
    return axo


def _ax_set_visible(ax, crosshair=None, xaxis=None, yaxis=None, xgrid=None, ygrid=None):
    if crosshair == False:
        ax.crosshair.hide()
    if xaxis is not None:
        ax.getAxis('bottom').setStyle(showValues=xaxis)
    if yaxis is not None:
        ax.getAxis('left').setStyle(showValues=yaxis)
    if xgrid is not None or ygrid is not None:
        ax.showGrid(x=xgrid, y=ygrid)
        if ax.getAxis('left'):
            ax.getAxis('left').setEnabled(False)
        if ax.getAxis('bottom'):
            ax.getAxis('bottom').setEnabled(False)


def _ax_decouple(ax):
    ax.setXLink(None)
    if ax.prev_ax:
        ax.prev_ax.set_visible(xaxis=True)


def _ax_disable_x_index(ax, decouple=True):
    ax.vb.x_indexed = False
    if decouple:
        _ax_decouple(ax)


def _ax_reset(ax):
    if ax.crosshair is not None:
        ax.crosshair.hide()
    for item in list(ax.items):
        ax.removeItem(item)
        if ax.vb.master_viewbox and hasattr(item, 'name') and item.name():
            legend = ax.vb.master_viewbox.parent().legend
            if legend:
                legend.removeItem(item)
    if ax.legend:
        ax.legend.opts['offset'] = None
        ax.legend.setParentItem(None)
        ax.legend = None
    ax.vb.reset()
    ax.vb.set_datasrc(None)
    if ax.crosshair is not None:
        ax.crosshair.show()


def _create_legend(ax):
    if ax.vb.master_viewbox:
        ax = ax.vb.master_viewbox.parent()
    if ax.legend is None:
        ax.legend = FinLegendItem(border_color=legend_border_color, fill_color=legend_fill_color, size=None, offset=(3,2))
        ax.legend.setParentItem(ax.vb)


def _update_significants(ax, datasrc, force):
    # check if no epsilon set yet
    default_dec = 0.99 < ax.significant_decimals/significant_decimals < 1.01
    default_eps = 0.99 < ax.significant_eps/significant_eps < 1.01
    if force or (default_dec and default_eps):
        try:
            sd,se = datasrc.calc_significant_decimals()
            if sd:
                if force or default_dec or sd > ax.significant_decimals:
                    ax.significant_decimals = sd
                if force or default_eps or se < ax.significant_eps:
                    ax.significant_eps = se
        except:
            pass # datasrc probably full av NaNs


def _is_standalone(timeser):
    # more than N percent gaps or time reversals probably means this is a standalone plot
    return timeser.isnull().sum() + (timeser.diff()<=0).sum() > len(timeser)*0.1


def _create_series(a):
    return a if isinstance(a, pd.Series) else pd.Series(a)


def _create_datasrc(ax, *args):
    def do_create(args):
        if len(args) == 1 and type(args[0]) == PandasDataSource:
            return args[0]
        if len(args) == 1 and type(args[0]) in (list, tuple):
            args = [np.array(args[0])]
        if len(args) == 1 and type(args[0]) == np.ndarray:
            args = [pd.DataFrame(args[0].T)]
        if len(args) == 1 and type(args[0]) == pd.DataFrame:
            return PandasDataSource(args[0])
        args = [_create_series(a) for a in args]
        return PandasDataSource(pd.concat(args, axis=1))
    iargs = [a for a in args if a is not None]
    datasrc = do_create(iargs)
    # check if time column missing
    if len(datasrc.df.columns) == 1:
        # assume time data has already been added before
        for a in ax.vb.win.axs:
            if a.vb.datasrc and len(a.vb.datasrc.df.columns) >= 2:
                datasrc.df.columns = a.vb.datasrc.df.columns[1:len(datasrc.df.columns)+1]
                col = a.vb.datasrc.df.columns[0]
                datasrc.df.insert(0, col, a.vb.datasrc.df[col])
                datasrc = PandasDataSource(datasrc.df)
                break
    elif len(iargs) >= 2 and len(datasrc.df.columns) == len(iargs)+1 and len(iargs) == len(args):
        try:
            if '.Int' in str(type(iargs[0].index)):
                print('WARNING: performance penalty and crash may occur when using int64 instead of range indices.')
                if (iargs[0].index == range(len(iargs[0]))).all():
                    print(' - Fix by .reset_index(drop=True)')
                    return _create_datasrc(ax, datasrc.df[datasrc.df.columns[1:]])
        except:
            print('WARNING: input data source may cause performance penalty and crash.')

    # FIX: stupid QT bug causes rectangles larger than 2G to flicker, so scale rendering down some
    if datasrc.df.iloc[:, 1:].max(numeric_only=True).max() > 1e8: # too close to 2G for comfort
        ax.vb.yscale.set_scale(int(1e8))
    return datasrc


def _set_datasrc(ax, datasrc, addcols=True):
    viewbox = ax.vb
    if not datasrc.standalone:
        if viewbox.datasrc is None:
            viewbox.set_datasrc(datasrc) # for mwheel zoom-scaling
            _set_x_limits(ax, datasrc)
        else:
            t0 = viewbox.datasrc.x.loc[0]
            if addcols:
                viewbox.datasrc.addcols(datasrc)
            else:
                viewbox.datasrc.df = datasrc.df
            # check if we need to re-render previous plots due to changed indices
            indices_updated = viewbox.datasrc.timebased() and t0 != viewbox.datasrc.x.loc[0]
            for item in ax.items:
                if hasattr(item, 'datasrc') and not item.datasrc.standalone:
                    item.datasrc.set_df(viewbox.datasrc.df) # every plot here now has the same time-frame
                    if indices_updated:
                        _start_visual_update(item)
                        _end_visual_update(item)
            _set_x_limits(ax, datasrc)
            viewbox.set_datasrc(viewbox.datasrc) # update zoom
    else:
        viewbox.standalones.add(datasrc)
        datasrc.update_init_x(viewbox.init_steps)
        ## if not viewbox.x_indexed:
            ## _set_x_limits(ax, datasrc)
    # update period if this datasrc has higher time resolution
    global epoch_period
    if datasrc.timebased() and (epoch_period > 1e7 or not datasrc.standalone):
        ep = datasrc.period
        epoch_period = ep if ep < epoch_period else epoch_period


def _has_timecol(df):
    return len(df.columns) >= 2


def _set_max_zoom(vbs):
    '''Set the relative allowed zoom level between axes groups, where the lowest-resolution
       plot in each group uses max_zoom_points, while the others get a scale >=1 of their
       respective highest zoom level.'''
    groups = defaultdict(set)
    for vb in vbs:
        master_vb = vb.linkedView(0)
        if vb.datasrc and master_vb and master_vb.datasrc:
            groups[master_vb].add(vb)
            groups[master_vb].add(master_vb)
    for group in groups.values():
        minlen = min(len(vb.datasrc.df) for vb in group)
        for vb in group:
            vb.max_zoom_points_f = len(vb.datasrc.df) / minlen


def _adjust_renko_datasrc(bins, step, datasrc):
    if not bins and not step:
        bins = 50
    if not step:
        step = (datasrc.y.max()-datasrc.y.min()) / bins
    bricks = datasrc.y.diff() / step
    bricks = (datasrc.y[bricks.isnull() | (bricks.abs()>=0.5)] / step).round().astype(int)
    extras = datasrc.df.iloc[:, datasrc.col_data_offset+1:]
    ts = datasrc.x[bricks.index]
    up = bricks.iloc[0] + 1
    dn = up - 2
    data = []
    for t,i,brick in zip(ts, bricks.index, bricks):
        s = 0
        if brick >= up:
            x0,x1,s = up-1,brick,+1
            up = brick+1
            dn = brick-2
        elif brick <= dn:
            x0,x1,s = dn,brick-1,-1
            up = brick+2
            dn = brick-1
        if s:
            for x in range(x0, x1, s):
                td = abs(x1-x)-1
                ds = 0 if s>0 else step
                y = x*step
                z = list(extras.loc[i])
                data.append([t-td, y+ds, y+step-ds, y+step, y] + z)
    datasrc.set_df(pd.DataFrame(data, columns='time open close high low'.split()+list(extras.columns)))


def _adjust_renko_log_datasrc(bins, step, datasrc):
    datasrc.df.iloc[:,1] = np.log10(datasrc.df.iloc[:,1])
    _adjust_renko_datasrc(bins, step, datasrc)
    datasrc.df.iloc[:,1:5] = 10**datasrc.df.iloc[:,1:5]


def _adjust_volume_datasrc(datasrc):
    if len(datasrc.df.columns) <= 4:
        datasrc.df.insert(3, '_zero_', [0]*len(datasrc.df)) # base of candles is always zero
    datasrc.set_df(datasrc.df.iloc[:,[0,3,4,1,2]]) # re-arrange columns for rendering
    datasrc.scale_cols = [1, 2] # scale by both baseline and volume


def _preadjust_horiz_datasrc(datasrc):
    arrayify = lambda d: d.values if type(d) == pd.DataFrame else d
    # create a dataframe from the input array
    times = [t for t,row in datasrc]
    if len(times) == 1: # add an empty time slot
        times = [times[0], times[0]+1]
        datasrc = datasrc + [[times[1], [(p,0) for p,v in arrayify(datasrc[0][1])]]]
    data = [[e for v in arrayify(row) for e in v] for t,row in datasrc]
    maxcols = max(len(row) for row in data)
    return pd.DataFrame(columns=range(maxcols), data=data, index=times)


def _adjust_horiz_datasrc(datasrc):
    # to be able to scale properly, move the last two values to the last two columns
    values = datasrc.df.iloc[:, 1:].values
    for i,nrow in enumerate(values):
        orow = nrow[~np.isnan(nrow)]
        if len(nrow) == len(orow) or len(orow) <= 2:
            continue
        nrow[-2:] = orow[-2:]
        nrow[len(orow)-2:len(orow)] = np.nan
    datasrc.df.iloc[:, 1:] = values


def _adjust_bar_datasrc(datasrc, order_cols=True):
    if len(datasrc.df.columns) <= 2:
        datasrc.df.insert(1, '_base_', [0]*len(datasrc.df)) # base
    if len(datasrc.df.columns) <= 4:
        datasrc.df.insert(1, '_open_',  [0]*len(datasrc.df)) # "open" for color
        datasrc.df.insert(2, '_close_', datasrc.df.iloc[:, 3]) # "close" (actual bar value) for color
    if order_cols:
        datasrc.set_df(datasrc.df.iloc[:,[0,3,4,1,2]]) # re-arrange columns for rendering
    datasrc.scale_cols = [1, 2] # scale by both baseline and volume


def _update_data(preadjustfunc, adjustfunc, item, ds, gfx=True):
    if preadjustfunc:
        ds = preadjustfunc(ds)
    ds = _create_datasrc(item.ax, ds)
    if adjustfunc:
        adjustfunc(ds)
    cs = list(item.datasrc.df.columns[:1]) + list(item.datasrc.df.columns[item.datasrc.col_data_offset:])
    if len(cs) >= len(ds.df.columns):
        ds.df.columns = cs[:len(ds.df.columns)]
    item.datasrc.update(ds)
    _set_datasrc(item.ax, item.datasrc, addcols=False)
    if gfx:
        item.update_gfx()


def _update_gfx(item):
    _start_visual_update(item)
    for i in item.ax.items:
        if i == item or not hasattr(i, 'datasrc'):
            continue
        lc = len(item.datasrc.df)
        li = len(i.datasrc.df)
        if lc and li and max(lc,li)/min(lc,li) > 100: # TODO: should be typed instead
            continue
        cc = item.datasrc.df.columns
        ci = i.datasrc.df.columns
        c0 = [c for c in ci if c in cc]
        c1 = [c for c in ci if not c in cc]
        df_clipped = item.datasrc.df[c0]
        if c1:
            df_clipped = df_clipped.copy()
            for c in c1:
                df_clipped[c] = i.datasrc.df[c]
        i.datasrc.set_df(df_clipped)
        break
    update_sigdig = False
    if not item.datasrc.standalone and not item.ax.vb.isMouseLeftDrag:
        # new limits when extending/reducing amount of data
        x_min,x1 = _set_x_limits(item.ax, item.datasrc)
        # scroll all plots if we're at the far right
        tr = item.ax.vb.targetRect()
        x0 = x1 - tr.width()
        if tr.right() < x1 - 5 - 2*right_margin_candles:
            x0 = x1 = None
        prev_top = item.ax.vb.targetRect().top()
        item.ax.vb.update_y_zoom(x0, x1)
        this_top = item.ax.vb.targetRect().top()
        if this_top and not (0.99 < abs(prev_top/this_top) < 1.01):
            update_sigdig = True
        if item.ax.axes['bottom']['item'].isVisible(): # update axes if visible
            item.ax.axes['bottom']['item'].hide()
            item.ax.axes['bottom']['item'].show()
    _end_visual_update(item)
    if update_sigdig:
        _update_significants(item.ax, item.ax.vb.datasrc, force=True)


def _start_visual_update(item):
    if isinstance(item, FinPlotItem):
        item.ax.removeItem(item)
        item.dirty = True
    else:
        item.setData(item.datasrc.index, item.datasrc.y)


def _end_visual_update(item):
    if isinstance(item, FinPlotItem):
        item.ax.addItem(item)
        item.repaint()


def _set_x_limits(ax, datasrc):
    x0,x1 = _xminmax(datasrc, x_indexed=ax.vb.x_indexed)
    ax.setLimits(xMin=x0, xMax=x1)
    return x0,x1


def _xminmax(datasrc, x_indexed, init_steps=None, extra_margin=0):
    if x_indexed and init_steps:
        # initial zoom
        x0 = max(datasrc.xlen-init_steps, 0) - side_margin - extra_margin
        x1 = datasrc.xlen - side_margin + extra_margin
    elif x_indexed:
        # total x size for indexed data
        x0 = -side_margin - extra_margin
        x1 = datasrc.xlen - 1 + side_margin + right_margin_candles + extra_margin# add another margin to get the "snap back" sensation
    else:
        # x size for plain Y-over-X data (i.e. not indexed)
        x0 = datasrc.x.min()
        x1 = datasrc.x.max()
        # extend margin on decoupled plots
        d = (x1-x0) * (0.2+extra_margin)
        x0 -= d
        x1 += d
    return x0,x1


def _repaint_candles():
    '''Candles are only partially drawn, and therefore needs manual dirty reminder whenever it goes off-screen.'''
    axs = [ax for win in windows for ax in win.axs] + overlay_axs
    for ax in axs:
        for item in list(ax.items):
            if isinstance(item, FinPlotItem):
                _start_visual_update(item)
                _end_visual_update(item)


def _paint_scatter(item, p, *args):
    with np.errstate(invalid='ignore'): # make pg's mask creation calls to numpy shut up
        item._dopaint(p, *args)


def _key_pressed(vb, ev):
    if ev.text() == 'g': # grid
        global clamp_grid
        clamp_grid = not clamp_grid
        for win in windows:
            for ax in win.axs:
                ax.crosshair.update()
    elif ev.text() in ('\r', ' '): # enter, space
        vb.set_draw_line_color(draw_done_color)
        vb.draw_line = None
    elif ev.text() in ('\x7f', '\b'): # del, backspace
        if not vb.remove_last_roi():
            return False
    elif ev.key() == QtCore.Qt.Key_Left:
        vb.pan_x(percent=-15)
    elif ev.key() == QtCore.Qt.Key_Right:
        vb.pan_x(percent=+15)
    elif ev.key() == QtCore.Qt.Key_Home:
        vb.pan_x(steps=-1e10)
        _repaint_candles()
    elif ev.key() == QtCore.Qt.Key_End:
        vb.pan_x(steps=+1e10)
        _repaint_candles()
    elif ev.key() == QtCore.Qt.Key_Escape:
        vb.win.close()
    else:
        return False
    return True


def _mouse_clicked(vb, ev):
    if ev.button() == 8: # back
        vb.pan_x(percent=-30)
    elif ev.button() == 16: # fwd
        vb.pan_x(percent=+30)
    else:
        return False
    return True


def _mouse_moved(master, evs):
    if hasattr(master, 'closing') and master.closing:
        return
    if not evs:
        evs = master_data[master]['last_mouse_evs']
        if not evs:
            return
    master_data[master]['last_mouse_evs'] = evs
    pos = evs[-1]
    # allow inter-pixel moves if moving mouse slowly
    y = pos.y()
    dy = y - master_data[master]['last_mouse_y']
    if 0 < abs(dy) <= 1:
        pos.setY(pos.y() - dy/2)
    master_data[master]['last_mouse_y'] = y
    # apply to all crosshairs
    for ax in master.axs:
        if ax.isVisible() and ax.crosshair:
            point = ax.vb.mapSceneToView(pos)
            ax.crosshair.update(point)


def _wheel_event_wrapper(self, orig_func, ev):
    # scrolling on the border is simply annoying, pop in a couple of pixels to make sure
    d = QtCore.QPoint(-2,0)
    ev = QtGui.QWheelEvent(ev.pos()+d, ev.globalPos()+d, ev.pixelDelta(), ev.angleDelta(), ev.angleDelta().y(), QtCore.Qt.Vertical, ev.buttons(), ev.modifiers())
    orig_func(self, ev)


def _inspect_clicked(ax, inspector, when_double_click, evs):
    if evs[-1].accepted or when_double_click != evs[-1].double():
        return
    pos = evs[-1].scenePos()
    return _inspect_pos(ax, inspector, (pos,))


def _inspect_pos(ax, inspector, poss):
    if not ax.vb.datasrc:
        return
    point = ax.vb.mapSceneToView(poss[-1])
    t = point.x() + 0.5
    try:
        t = ax.vb.datasrc.closest_time(t)
    except KeyError: # when clicking beyond right_margin_candles
        if clamp_grid:
            t = ax.vb.datasrc.x.iloc[-1 if t > 0 else 0]
    try:
        inspector(t, point.y())
    except OSError as e:
        pass
    except Exception as e:
        print('Inspection error:', type(e), e)


def brighten(color, f):
    if not color:
        return color
    return pg.mkColor(color).lighter(int(f*100))


def _get_color(ax, style, wanted_color):
    if type(wanted_color) in (str, QtGui.QColor):
        return wanted_color
    index = wanted_color if type(wanted_color) == int else None
    is_line = lambda style: style is None or any(ch in style for ch in '-_.')
    this_line = is_line(style)
    if this_line:
        colors = soft_colors
    else:
        colors = hard_colors
    if index is None:
        avoid = set(i.opts['handed_color'] for i in ax.items if isinstance(i,pg.PlotDataItem) and i.opts['handed_color'] is not None and this_line==is_line(i.opts['symbol']))
        index = len([i for i in ax.items if isinstance(i,pg.PlotDataItem) and i.opts['handed_color'] is None and this_line==is_line(i.opts['symbol'])])
        while index in avoid:
            index += 1
    return colors[index%len(colors)]


def _pdtime2epoch(t):
    if isinstance(t, pd.Series):
        if isinstance(t.iloc[0], pd.Timestamp):
            return t.astype('int64')
        h = np.nanmax(t.values)
        if h < 1e10: # handle s epochs
            return (t*1e9).astype('int64')
        if h < 1e13: # handle ns epochs
            return (t*1e6).astype('int64')
        if h < 1e16: # handle us epochs
            return (t*1e3).astype('int64')
        return t.astype('int64')
    return t


def _pdtime2index(ax, ts, any_end=False, require_time=False):
    if isinstance(ts.iloc[0], pd.Timestamp):
        ts = ts.astype('int64')
    else:
        h = np.nanmax(ts.values)
        if h < 1e7:
            if require_time:
                assert False, 'not a time series'
            return ts
        if h < 1e10: # handle s epochs
            ts = ts.astype('float64') * 1e9
        elif h < 1e13: # handle ms epochs
            ts = ts.astype('float64') * 1e6
        elif h < 1e16: # handle us epochs
            ts = ts.astype('float64') * 1e3
    
    datasrc = _get_datasrc(ax)
    xs = datasrc.x

    # try exact match before approximate match
    exact = datasrc.index[xs.isin(ts)].to_list()
    if len(exact) == len(ts):
        return exact
    
    r = []
    for i,t in enumerate(ts):
        xss = xs.loc[xs>t]
        if len(xss) == 0:
            t0 = xs.iloc[-1]
            if any_end or t0 == t:
                r.append(len(xs)-1)
                continue
            if i > 0:
                continue
            assert t <= t0, 'must plot this primitive in prior time-range'
        i1 = xss.index[0]
        i0 = i1-1
        if i0 < 0:
            i0,i1 = 0,1
        t0,t1 = xs.loc[i0], xs.loc[i1]
        dt = (t-t0) / (t1-t0)
        r.append(lerp(dt, i0, i1))
    return r


def _get_datasrc(ax, require=True):
    if ax.vb.datasrc is not None or not ax.vb.x_indexed:
        return ax.vb.datasrc
    vbs = [ax.vb for win in windows for ax in win.axs]
    for vb in vbs:
        if vb.datasrc:
            return vb.datasrc
    if require:
        assert ax.vb.datasrc, 'not possible to plot this primitive without a prior time-range to compare to'


def _millisecond_wrap(s):
    return (s+'.000000') if '.' not in s else s


def _x2local_t(datasrc, x):
    if display_timezone == None:
        return _x2utc(datasrc, x)
    return _x2t(datasrc, x, lambda t: _millisecond_wrap(datetime.fromtimestamp(t/1e9, tz=display_timezone).isoformat(sep=' ').partition('+')[0]))


def _x2utc(datasrc, x):
    # using pd.to_datetime allow for pre-1970 dates
    return _x2t(datasrc, x, lambda t: pd.to_datetime(t, unit='ns').strftime('%Y-%m-%d %H:%M:%S.%f'))


def _x2t(datasrc, x, ts2str):
    if not datasrc:
        return '',False
    try:
        x += 0.5
        t,_,_,_,cnt = datasrc.hilo(x, x)
        if cnt:
            if not datasrc.timebased():
                return '%g' % t, False
            s = ts2str(t)
            
            if epoch_period >= 23*60*60: # daylight savings, leap seconds, etc
                i = s.index(' ')
            elif epoch_period >= 59: # consider leap seconds
                i = s.rindex(':')
            elif epoch_period >= 1:
                i = s.index('.') if '.' in s else len(s)
            elif epoch_period >= 0.001:
                i = -3
            else:
                i = len(s)
            return s[:i],True
    except Exception as e:
        import traceback
        traceback.print_exc()
    return '',datasrc.timebased()


def _x2year(datasrc, x):
    t,hasds = _x2local_t(datasrc, x)
    return t[:4],hasds


def _round_to_significant(rng, rngmax, x, significant_decimals, significant_eps):
    is_highres = (rng/significant_eps > 1e2 and rngmax<1e-2) or abs(rngmax) > 1e7 or rng < 1e-5
    sd = significant_decimals
    if is_highres and abs(x)>0:
        exp10 = floor(np.log10(abs(x)))
        x = x / (10**exp10)
        rm = int(abs(np.log10(rngmax))) if rngmax>0 else 0
        sd = min(3, sd+rm)
        fmt = '%%%i.%ife%%i' % (sd, sd)
        r = fmt % (x, exp10)
    else:
        eps = fmod(x, significant_eps)
        if abs(eps) >= significant_eps/2:
            # round up
            eps -= np.sign(eps)*significant_eps
        x -= eps
        fmt = '%%%i.%if' % (sd, sd)
        r = fmt % x
    return r


def _roihandle_move_snap(vb, orig_func, pos, modifiers=QtCore.Qt.KeyboardModifier(), finish=True):
    pos = vb.mapDeviceToView(pos)
    pos = _clamp_point(vb.parent(), pos)
    pos = vb.mapViewToDevice(pos)
    orig_func(pos, modifiers=modifiers, finish=finish)


def _clamp_xy(ax, x, y):
    y = ax.vb.yscale.xform(y)
    if clamp_grid and ax.vb.x_indexed:
        ds = ax.vb.datasrc
        if x < 0 or (ds and x > len(ds.df)-1):
            x = 0 if x < 0 else len(ds.df)-1
        x = _round(x)
        eps = ax.significant_eps
        if eps > 1e-8:
            eps2 = np.sign(y) * 0.5 * eps
            y -= fmod(y+eps2, eps) - eps2
    y = ax.vb.yscale.invxform(y, verify=True)
    return x, y


def _clamp_point(ax, p):
    if clamp_grid:
        x,y = _clamp_xy(ax, p.x(), p.y())
        return pg.Point(x, y)
    return p


def _draw_line_segment_text(polyline, segment, pos0, pos1):
    fsecs = None
    datasrc = polyline.vb.datasrc
    if datasrc and clamp_grid:
        try:
            x0,x1 = pos0.x()+0.5, pos1.x()+0.5
            t0,_,_,_,cnt0 = datasrc.hilo(x0, x0)
            t1,_,_,_,cnt1 = datasrc.hilo(x1, x1)
            if cnt0 and cnt1:
                fsecs = abs(t1 - t0) / 1e9
        except:
            pass
    diff = pos1 - pos0
    if fsecs is None:
        fsecs = abs(diff.x()*epoch_period)
    secs = int(fsecs)
    mins = secs//60
    hours = mins//60
    mins = mins%60
    secs = secs%60
    if hours==0 and mins==0 and secs < 60 and epoch_period < 1:
        msecs = int((fsecs-int(fsecs))*1000)
        ts = '%0.2i:%0.2i.%0.3i' % (mins, secs, msecs)
    elif hours==0 and mins < 60 and epoch_period < 60:
        ts = '%0.2i:%0.2i:%0.2i' % (hours, mins, secs)
    elif hours < 24:
        ts = '%0.2i:%0.2i' % (hours, mins)
    else:
        days = hours // 24
        hours %= 24
        ts = '%id %0.2i:%0.2i' % (days, hours, mins)
        if ts.endswith(' 00:00'):
            ts = ts.partition(' ')[0]
    ysc = polyline.vb.yscale
    if polyline.vb.y_positive:
        y0,y1 = ysc.xform(pos0.y()), ysc.xform(pos1.y())
        if y0:
            value = '%+.2f %%' % (100 * y1 / y0 - 100)
        elif not y1:
            value = '0'
        else:
            value = '+∞' if y1>0 else '-∞'
    else:
        dy = ysc.xform(diff.y())
        if dy and (abs(dy) >= 1e4 or abs(dy) <= 1e-2):
            value = '%+3.3g' % dy
        else:
            value = '%+2.2f' % dy
    extra = _draw_line_extra_text(polyline, segment, pos0, pos1)
    return '%s %s (%s)' % (value, extra, ts)


def _draw_line_extra_text(polyline, segment, pos0, pos1):
    '''Shows the proportions of this line height compared to the previous segment.'''
    prev_text = None
    for text in polyline.texts:
        if prev_text is not None and text.segment == segment:
            h0 = prev_text.segment.handles[0]['item']
            h1 = prev_text.segment.handles[1]['item']
            prev_change = h1.pos().y() - h0.pos().y()
            this_change = pos1.y() - pos0.y()
            if not abs(prev_change) > 1e-14:
                break
            change_part = abs(this_change / prev_change)
            return ' = 1:%.2f ' % change_part
        prev_text = text
    return ''


def _makepen(color, style=None, width=1):
    if style is None or style == '-':
        return pg.mkPen(color=color, width=width)
    dash = []
    for ch in style:
        if ch == '-':
            dash += [4,2]
        elif ch == '_':
            dash += [10,2]
        elif ch == '.':
            dash += [1,2]
        elif ch == ' ':
            if dash:
                dash[-1] += 2
    return pg.mkPen(color=color, style=QtCore.Qt.CustomDashLine, dash=dash, width=width)


def _round(v):
    return ceil(v-0.5)


try:
    qtver = '%d.%d' % (QtCore.QT_VERSION//256//256, QtCore.QT_VERSION//256%256)
    if qtver not in ('5.9', '5.13') and [int(i) for i in pg.__version__.split('.')] <= [0,11,0]:
        print('WARNING: your version of Qt may not plot curves containing NaNs and is not recommended.')
        print('See https://github.com/pyqtgraph/pyqtgraph/issues/1057')
except:
    pass


# default to black-on-white
pg.widgets.GraphicsView.GraphicsView.wheelEvent = partialmethod(_wheel_event_wrapper, pg.widgets.GraphicsView.GraphicsView.wheelEvent)
# use finplot instead of matplotlib
pd.set_option('plotting.backend', 'finplot.pdplot')
# pick up win resolution
try:
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    lod_candles = int(user32.GetSystemMetrics(0) * 1.6)
    candle_shadow_width = int(user32.GetSystemMetrics(0) // 2100 + 1) # 2560 and resolutions above -> wider shadows
except:
    pass


if False: # performance measurement code
    import time, sys
    def self_timecall(self, pname, fname, func, *args, **kwargs):
        ## print('self_timecall', pname, fname)
        t0 = time.perf_counter()
        r = func(self, *args, **kwargs)
        t1 = time.perf_counter()
        print('%s.%s: %f' % (pname, fname, t1-t0))
        return r
    def timecall(fname, func, *args, **kwargs):
        ## print('timecall', fname)
        t0 = time.perf_counter()
        r = func(*args, **kwargs)
        t1 = time.perf_counter()
        print('%s: %f' % (fname, t1-t0))
        return r
    def wrappable(fn, f):
        try:    return callable(f) and str(f.__module__) == 'finplot'
        except: return False
    m = sys.modules['finplot']
    for fname in dir(m):
        func = getattr(m, fname)
        if wrappable(fname, func):
            for fname2 in dir(func):
                func2 = getattr(func, fname2)
                if wrappable(fname2, func2):
                    print(fname, str(type(func)), '->', fname2, str(type(func2)))
                    setattr(func, fname2, partialmethod(self_timecall, fname, fname2, func2))
            setattr(m, fname, partial(timecall, fname, func))
