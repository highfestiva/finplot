from math import ceil, floor

import numpy as np
import pandas as pd
import pyqtgraph as pg

from FP_Tools import _makepen
from FP_Setting import time_splits
from FP_Time_Tools import display_timezone, _x2year, _x2local_t, _get_datasrc, _pdtime2index


class EpochAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb

    def tickStrings(self, values, scale, spacing):
        if self.mode == 'num':
            return ['%g'%v for v in values]
        conv = _x2year if self.mode=='years' else _x2local_t
        strs = [conv(self.vb.datasrc, value)[0] for value in values]
        if all(s.endswith(' 00:00') for s in strs if s): # all at midnight -> round to days
            strs = [s.partition(' ')[0] for s in strs]
        return strs

    def tickValues(self, minVal, maxVal, size):
        self.mode = 'num'
        ax = self.vb.parent()
        datasrc = _get_datasrc(ax, require=False)
        if datasrc is None or not self.vb.x_indexed:
            return super().tickValues(minVal, maxVal, size)
        # calculate if we use years, days, etc.
        t0,t1,_,_,_ = datasrc.hilo(minVal, maxVal)
        t0,t1 = pd.to_datetime(t0), pd.to_datetime(t1)
        dts = (t1-t0).total_seconds()
        gfx_width = int(size)
        for mode, dtt, freq, ticklen in time_splits:
            if dts > dtt:
                self.mode = mode
                desired_ticks = gfx_width / ((ticklen+2) * 10) - 1 # an approximation is fine
                if self.vb.datasrc is not None and not self.vb.datasrc.is_smooth_time():
                    desired_ticks -= 1 # leave more space for unevenly spaced ticks
                desired_ticks = max(desired_ticks, 4)
                to_midnight = freq in ('YS','MS', 'W-MON', 'D')
                tz = display_timezone if to_midnight else None # for shorter timeframes, timezone seems buggy
                rng = pd.date_range(t0, t1, tz=tz, normalize=to_midnight, freq=freq)
                steps = len(rng) if len(rng)&1==0 else len(rng)+1 # reduce jitter between e.g. 5<-->10 ticks for resolution close to limit
                step = int(steps/desired_ticks) or 1
                rng = rng[::step]
                if not to_midnight:
                    try:    rng = rng.round(freq=freq)
                    except: pass
                ax = self.vb.parent()
                rng = _pdtime2index(ax=ax, ts=pd.Series(rng), require_time=True)
                indices = [ceil(i) for i in rng]
                return [(0, indices)]
        return [(0,[])]

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs:
            if not self.style['showValues']:
                pen,p0,p1 = specs[0] # axis specs
                specs = [(_makepen('#fff0'),p0,p1)] + list(specs[1:]) # don't draw axis if hiding values
            else:
                # throw out ticks that are out of bounds
                text_specs = specs[2]
                if len(text_specs) >= 4:
                    rect,flags,text = text_specs[0]
                    if rect.left() < 0:
                        del text_specs[0]
                    rect,flags,text = text_specs[-1]
                    if rect.right() > self.geometry().width():
                        del text_specs[-1]
                # ... and those that overlap
                x = 1e6
                for i,(rect,flags,text) in reversed(list(enumerate(text_specs))):
                    if rect.right() >= x:
                        del text_specs[i]
                    else:
                        x = rect.left()
        return specs


class YAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb
        self.hide_strings = False
        self.style['autoExpandTextSpace'] = False
        self.style['autoReduceTextSpace'] = False
        self.next_fmt = '%g'

    def tickValues(self, minVal, maxVal, size):
        vs = super().tickValues(minVal, maxVal, size)
        if len(vs) < 3:
            return vs
        return self.fmt_values(vs)

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        v1 = int(floor(minVal))
        v2 = int(ceil(maxVal))
        minor = []
        for v in range(v1, v2):
            minor.extend([v+l for l in np.log10(np.linspace(1, 9.9, 90))])
        minor = [x for x in minor if x>minVal and x<maxVal]
        if len(minor) > 10:
            minor = minor[::len(minor)//5]
        vs = [(None, minor)]
        return self.fmt_values(vs)

    def tickStrings(self, values, scale, spacing):
        if self.hide_strings:
            return []
        xform = self.vb.yscale.xform
        return [self.next_fmt%xform(value) for value in values]

    def fmt_values(self, vs):
        xform = self.vb.yscale.xform
        gs = ['%g'%xform(v) for v in vs[-1][1]]
        if any(['e' in g for g in gs]):
            maxdec = max([len((g).partition('.')[2].partition('e')[0]) for g in gs if 'e' in g])
            self.next_fmt = '%%.%ie' % maxdec
        else:
            maxdec = max([len((g).partition('.')[2]) for g in gs])
            self.next_fmt = '%%.%if' % maxdec
        return vs


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
                return -1e6 / self.scalef
            y = np.log10(y)
        return y
