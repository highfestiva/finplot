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

finplot *is not a web app*. It does not help you create an homebrew exchange. It does not work with Jupyter Labs.

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

For more examples and a bunch of snippets, see the [examples](https://github.com/highfestiva/finplot/blob/master/finplot/examples/)
directory or the [wiki](https://github.com/highfestiva/finplot/wiki). There you'll find how to plot MACD, Parabolic SAR, RSI,
volume profile and much more.


## Coffee

For future support and features, consider a small donation.

BTC: bc1qk8m8yh86l2pz4eypflchr0tkn5aeud6cmt426m

ETH: 0x684d7d4C52ed428AE9a36B2407ba909D896cDB67
