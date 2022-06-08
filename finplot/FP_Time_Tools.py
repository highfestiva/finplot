from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.tz import tzlocal

display_timezone = tzlocal() # default to local
windows = [] # no gc
epoch_period = 1e30
lerp = lambda t,a,b: t*b+(1-t)*a


def _pdtime2epoch(t):
    if isinstance(t, pd.Series):
        if isinstance(t.iloc[0], pd.Timestamp):
            return t.view('int64')
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
        ts = ts.view('int64')
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


def _millisecond_tz_wrap(s):
    if len(s) > 6 and s[-6] in '+-' and s[-3] == ':': # +01:00 fmt timezone present?
        s = s[:-6]
    return (s+'.000000') if '.' not in s else s


def _x2local_t(datasrc, x):
    if display_timezone == None:
        return _x2utc(datasrc, x)
    return _x2t(datasrc, x, lambda t: _millisecond_tz_wrap(datetime.fromtimestamp(t/1e9, tz=display_timezone).isoformat(sep=' ')))


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
