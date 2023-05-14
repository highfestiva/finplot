'''
Used as Pandas plotting backend.
'''

import finplot


def plot(df, x, y, kind, **kwargs):
    _x = df.index if y is None else df[x]
    try:
        _y = df[x].reset_index(drop=True) if y is None else df[y]
    except:
        _y = df.reset_index(drop=True)
    kwargs = dict(kwargs)
    if 'by' in kwargs:
        del kwargs['by']
    if kind in ('candle', 'candle_ochl', 'candlestick', 'candlestick_ochl', 'volume', 'volume_ocv', 'renko'):
        if 'candle' in kind:
            return finplot.candlestick_ochl(df, **kwargs)
        elif 'volume' in kind:
            return finplot.volume_ocv(df, **kwargs)
        elif 'renko' in kind:
            return finplot.renko(df, **kwargs)
    elif kind == 'scatter':
        if 'style' not in kwargs:
            kwargs['style'] = 'o'
        if type(x) is str and type(y) is str and _x is not None and _y is not None:
            return finplot.plot(_x, _y, **kwargs)
        else:
            return finplot.plot(df, **kwargs)
    elif kind == 'bar':
        if type(x) is str and type(y) is str and _x is not None and _y is not None:
            return finplot.bar(_x, _y, **kwargs)
        else:
            return finplot.bar(df, **kwargs)
    elif kind in ('barh', 'horiz_time_volume'):
        return finplot.horiz_time_volume(df, **kwargs)
    elif kind in ('heatmap'):
        return finplot.heatmap(df, **kwargs)
    elif kind in ('labels'):
        if type(x) is str and type(y) is str and _x is not None and _y is not None:
            return finplot.labels(_x, _y, **kwargs)
        else:
            return finplot.labels(df, **kwargs)
    elif kind in ('hist', 'histogram'):
        return finplot.hist(df, **kwargs)
    else:
        if x is None:
            _x = df
            _y = None
        if 'style' not in kwargs:
            kwargs['style'] = None
        return finplot.plot(_x, _y, **kwargs)
