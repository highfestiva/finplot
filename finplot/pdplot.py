import finplot


def plot(df, x, y, kind, **kwargs):
    _x = df.index if y is None else df[x]
    try:
        _y = df[x].reset_index(drop=True) if y is None else df[y]
    except:
        _y = df.reset_index(drop=True)
    if kind in ('candle', 'candle_ochl', 'candlestick', 'candlestick_ochl', 'volume', 'volume_ocv', 'renko'):
        if 'candle' in kind:
            return finplot.candlestick_ochl(df, **kwargs)
        elif 'volume' in kind:
            return finplot.volume_ocv(df, **kwargs)
        elif 'renko' in kind:
            return finplot.renko(_x, _y, **kwargs)
    elif kind == 'scatter':
        return finplot.plot(_x, _y, style='o', **kwargs)
    elif kind == 'bar':
        return finplot.bar(_x, _y, **kwargs)
    elif kind in ('barh', 'horiz_time_volume'):
        return finplot.horiz_time_volume(df, **kwargs)
    elif kind in ('heatmap'):
        return finplot.heatmap(df, **kwargs)
    elif kind in ('labels'):
        return finplot.labels(df, **kwargs)
    elif kind == 'hist':
        _y = y if y else x
        return finplot.hist(_x, _y, **kwargs)
    else:
        return finplot.plot(_x, _y, style=None, **kwargs)
