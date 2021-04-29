# Finance Plot

Finance Plotter, or finplot, is a performant library with a clean api to help you with your backtesting. It's
optionated with good defaults, so you can start doing your work without having to setup plots, colors, scales,
autoscaling, keybindings, handle panning+vertical zooming (which all non-finance libraries have problems with).
And best of all: it can show hundreds of thousands of datapoints without batting an eye.

<img src="https://badge.fury.io/py/finplot.svg"/> <img src="https://pepy.tech/badge/finplot/month"/> <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>


## Features

* Great performance compared to mpl_finance, plotly and Bokeh
* Clean api
* Works with both stocks as well as cryptocurrencies on any time resolution
* Show as many charts as you want on the same time axis, zoom on all of them at once
* Auto-reload position where you were looking last run
* Overlays, fill between, value bands, symbols, labels, legend, volume profile, heatmaps, etc.
* Can show real-time updates, including orderbook. Save screenshot.
* Comes with a [dozen](https://github.com/highfestiva/finplot/blob/master/finplot/examples) great examples.

![feature1](https://raw.githubusercontent.com/highfestiva/finplot/master/feature1.png)

![feature2](https://raw.githubusercontent.com/highfestiva/finplot/master/feature2.jpg)

![feature3](https://raw.githubusercontent.com/highfestiva/finplot/master/feature3.jpg)

![feature3](https://raw.githubusercontent.com/highfestiva/finplot/master/feature-nuts.jpg)


## What it is not

finplot is not a web app. It does not help you create an homebrew exchange. It does not work with Jupyter Labs.

It is only intended for you to do backtesting in. That is not to say that you can't create a ticker or a trade
widget yourself. The library is based on the eminent pyqtgraph, which is fast and flexible, so feel free to hack
away if that's what you want.


## Easy installation

```bash
$ pip install finplot
```


## Example

It's straight-forward to start using. This shows every daily candle of Apple since the 80'ies:

```python
import finplot as fplt
import yfinance

df = yfinance.download('AAPL')
fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
fplt.show()
```


## Example 2

![sample](https://raw.githubusercontent.com/highfestiva/finplot/master/screenshot.jpg)


This 25-liner pulls some BitCoin data off of Bittrex and shows the above:


```python
import finplot as fplt
import numpy as np
import pandas as pd
import requests

# pull some data
symbol = 'USDT-BTC'
url = 'https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName=%s&tickInterval=fiveMin' % symbol
data = requests.get(url).json()

# format it in pandas
df = pd.DataFrame(data['result'])
df = df.rename(columns={'T':'time', 'O':'open', 'C':'close', 'H':'high', 'L':'low', 'V':'volume'})
df = df.astype({'time':'datetime64[ns]'})

# create two axes
ax,ax2 = fplt.create_plot(symbol, rows=2)

# plot candle sticks
candles = df[['time','open','close','high','low']]
fplt.candlestick_ochl(candles, ax=ax)

# overlay volume on the top plot
volumes = df[['time','open','close','volume']]
fplt.volume_ocv(volumes, ax=ax.overlay())

# put an MA on the close price
fplt.plot(df['time'], df['close'].rolling(25).mean(), ax=ax, legend='ma-25')

# place some dumb markers on low wicks
lo_wicks = df[['open','close']].T.min() - df['low']
df.loc[(lo_wicks>lo_wicks.quantile(0.99)), 'marker'] = df['low']
fplt.plot(df['time'], df['marker'], ax=ax, color='#4a5', style='^', legend='dumb mark')

# draw some random crap on our second plot
fplt.plot(df['time'], np.random.normal(size=len(df)), ax=ax2, color='#927', legend='stuff')
fplt.set_y_range(-1.4, +3.7, ax=ax2) # hard-code y-axis range limitation

# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()

# we're done
fplt.show()
```


## Real-time examples

Included in this repo are [a 40-liner Bitfinex example](https://github.com/highfestiva/finplot/blob/master/finplot/examples/bfx.py)
and [a slightly longer BitMEX websocket example](https://github.com/highfestiva/finplot/blob/master/finplot/examples/bitmex-ws.py),
which both update in realtime with Bitcoin/Dollar pulled from the exchange.
[A more complicated example](https://github.com/highfestiva/finplot/blob/master/finplot/examples/complicated.py) show real-time
updates and interactively varying of asset, time scales, indicators and color scheme.

finplot is mainly intended for backtesting, so the API is clunky for real-time applications. The
[examples/complicated.py](https://github.com/highfestiva/finplot/blob/master/finplot/examples/complicated.py) was written a result
of popular demand.


## MACD, Parabolic SAR, RSI, volume profile and others

There are plenty of examples that show different indicators.

| Indicator | Example |
|-----------|---------|
| MACD | [S&P 500](https://github.com/highfestiva/finplot/blob/master/finplot/examples/snp500.py) |
| RSI | [Analyze](https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze.py) |
| SMA | [Analyze 2](https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze-2.py) |
| EMA | [Analyze](https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze.py) |
| TD sequential | [Bitfinex](https://github.com/highfestiva/finplot/blob/master/finplot/examples/bfx.py) |
| Bollinger bands | [BitMEX](https://github.com/highfestiva/finplot/blob/master/finplot/examples/bitmex-ws.py) |
| Parabolic SAR | [BitMEX](https://github.com/highfestiva/finplot/blob/master/finplot/examples/complicated.py) |
| Heikin ashi | [Analyze](https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze.py) |
| Renko | [Renko dark mode](https://github.com/highfestiva/finplot/blob/master/finplot/examples/renko-dark-mode.py) |
| Accumulation/distribution | [Analyze](https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze.py) |
| On balance volume | [Analyze](https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze.py) |
| Heat map | [Heatmap](https://github.com/highfestiva/finplot/blob/master/finplot/examples/heatmap.py) |
| Volume profile | [Volume profile](https://github.com/highfestiva/finplot/blob/master/finplot/examples/volume-profile.py) |
| VWAP | [Volume profile](https://github.com/highfestiva/finplot/blob/master/finplot/examples/volume-profile.py) |
| Period returns | [Analyze 2](https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze-2.py) |
| Asset correlation | [Overlay correlate](https://github.com/highfestiva/finplot/blob/master/finplot/examples/overlay-correlate.py) |
| Lines | [Bitcoin long term](https://github.com/highfestiva/finplot/blob/master/finplot/examples/btc-long-term.py) |
| ms time resolution | [Line](https://github.com/highfestiva/finplot/blob/master/finplot/examples/line.py) |

For interactively modifying what indicators are shown, see
[examples/complicated.py](https://github.com/highfestiva/finplot/blob/master/finplot/examples/complicated.py).


## Snippets

### Background color
```python
# finplot uses no background (i.e. white) on even rows and a slightly different color on odd rows.
# Set your own before creating the plot.
fplt.background = '#ff0' # yellow
fplt.odd_plot_background = '#f0f' # purple
fplt.plot(df.Close)
fplt.show()
```

### Unordered time series
finplot requires time-ordered time series - otherwise you'll get a crosshair and an X-axis showing the
millisecond epoch instead of the actual time. See my comment
[here](https://github.com/highfestiva/finplot/issues/58#issuecomment-716054127) and
[issue 50](https://github.com/highfestiva/finplot/issues/50) for more info.

It is also imperative that you either put your datetimes in your index, or in the first column. If your
datetime is in the first column, you normally want to have a zero-based range index,
`df.reset_index(drop=True)`, before plotting.

### Restore the zoom at startup
```python
# By default finplot shows all or a subset of your time series at startup. To store/restore zoom position:
fplt.autoviewrestore()
fplt.show() # will load zoom when showing, and save zoom when closing
```

### Time zone
```python
# Pandas normally reads datetimes in UTC time zone.
# finplot by default use the local time zone of your computer (for crosshair and X-axis)
from dateutil.tz import gettz
fplt.display_timezone = gettz('Asia/Jakarta')

# ... or in UTC = "display same as timezone-unaware data"
import datetime
finplot.display_timezone = datetime.timezone.utc
```

### Scatter plot with X-offset
To offset your scatter markers (say 0.2 time intervals to the left), see my comment
[here](https://github.com/highfestiva/finplot/issues/31#issuecomment-695952455).

### Align X-axes
See [issue 27](https://github.com/highfestiva/finplot/issues/27), and possibly (rarely a problem)
[issue 4](https://github.com/highfestiva/finplot/issues/4).

### Disable zoom/pan sync between axes
```python
# finplot assumes all your axes are in the same time span. To decouple the zoom/pan link, use:
ax2.decouple()
```

### Move viewport along X-axis (and autozoom)
Use `fplt.set_x_pos(xmin, xmax, ax)`. See
[examples/animate.py](https://github.com/highfestiva/finplot/blob/master/finplot/examples/animate.py).

### Place Region of Interest (ROI) markers
For placing ellipses, see [issue 57](https://github.com/highfestiva/finplot/issues/57).
For drawing lines, see [examples/line.py](https://github.com/highfestiva/finplot/blob/master/finplot/examples/line.py).
(Interactively use Ctrl+drag for lines and Ctrl+mbutton-drag for ellipses.)

### More than one Y-axis in same viewbox
```python
fplt.candlestick_ochl(df2[['Open','Close','High','Low']], ax=ax.overlay(scale=1.0, yaxis='linear'))
```
The `scale` parameter means it goes all the way to the top of the axis (volume normally stays at the bottom).
The `yaxis` parameter can be one of `False` (hidden which is default), `'linear'` or `'log'`.
See [issue 52](https://github.com/highfestiva/finplot/issues/52) for more info.

### Plot non-timeseries
finplot is made for plotting time series. To plot something different use `ax.disable_x_index()`. See second
axis of [examples/overlay-correlate.py](https://github.com/highfestiva/finplot/blob/master/finplot/examples/overlay-correlate.py).

### Custom crosshair and legend
[S&P500 example](https://github.com/highfestiva/finplot/blob/master/finplot/examples/snp500.py) shows how
to set crosshair texts and update legend text+color as a result of mouse hover.

### Custom axes ticks
To use your own labels on the X-axis see [comment on issue 50](https://github.com/highfestiva/finplot/issues/50#issuecomment-707929546).
If you want to roll your own Y-axis, inherit `fplt.YAxisItem`.

### Saving screenshot
See [examples/line.py](https://github.com/highfestiva/finplot/blob/master/finplot/examples/line.py).
To keep screenshot in RAM see [issue 28](https://github.com/highfestiva/finplot/issues/28).

For creating multiple screenshots see [issue 71](https://github.com/highfestiva/finplot/issues/71#issuecomment-742015927).

### Scaling plot heights
See [issue 56](https://github.com/highfestiva/finplot/issues/56). Changing the default window size can be
achieved by setting `fplt.winw = 900; fplt.winh = 500;` before creating your plot.

### Threading
See [issue 55](https://github.com/highfestiva/finplot/issues/55).

### Titles on axes
See [issue 41](https://github.com/highfestiva/finplot/issues/41). To show grid and further adapt axes, etc:

```python
ax.set_visible(crosshair=False, xaxis=False, yaxis=True, xgrid=True, ygrid=True)
```

### Fixing auto-zoom on realtime updates
See [issue 131](https://github.com/highfestiva/finplot/issues/131#issuecomment-786245998).

### Beep
```python
fplt.play_sound('bot-happy.wav') # Ooh! Watch me - I just made a profit!
```

### Keys
`Esc`, `Home`, `End`, `g`, `Left arrow`, `Right arrow`. `Ctrl+drag`.

### Missing snippets
Plot valign on mouse hover, update an orderbook, etc.


## Coffee

For future support and features, consider a small donation.

BTC: bc1qk8m8yh86l2pz4eypflchr0tkn5aeud6cmt426m

ETH: 0x684d7d4C52ed428AE9a36B2407ba909D896cDB67
