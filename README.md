# Finance Plot
Plotting financial data in Python for the most part sucks. [mpl_finance](https://github.com/matplotlib/mpl_finance) is horribly slow --
and plotly too -- when displaying 100k candlesticks or more.

Meet Finance Plotter, or finplot! It utilizes pyqtgraph underneath, which to my surprise was an excellent library. Finance Plot is built
to immitate matplotlib's api, but with better default looks and much better performance.

Also included is some zooming+scaling logic, so you won't get that "panning all over the place" feeling as one normally finds in
most non-finance plot libraries.


## Easy installation

You will need to pip install a couple of libraries, but no external installations required. (Which is a nice change under Windows, as
pretty much everything performant used to require MSVC altercation or external installation of binaries.)

So all you need to do (at least in Win10/Py3.5.1x64 2018-08) is:

```bash
$ pip install pandas
$ pip install pyqtgraph
$ pip install PyQT5==5.9 # for some reason 5.11 did not work
```


## Example

![sample](https://raw.githubusercontent.com/highfestiva/finplot/master/screenshot.jpg)


This 25-liner pulls some BitCoin data and shows the above:


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

# create three plots
ax,ax2,ax3 = fplt.create_plot(symbol, rows=3)

# plot candle sticks
candle_src = fplt.PandasDataSource(df[['time','open','close','high','low']])
fplt.candlestick_ochl(candle_src, ax=ax)

# put an MA in there
fplt.plot(df['time'], df['close'].rolling(25).mean(), ax=ax, color='#0000ff', legend='ma-25')

# place some dumb markers
hi_wicks = df['high'] - df[['open','close']].T.max().T
df.loc[(hi_wicks>hi_wicks.quantile(0.99)), 'marker'] = df['close']
fplt.plot(df['time'], df['marker'], ax=ax, color='#000000', style='^', legend='dumb mark')

# draw some random crap on our second plot
df['rnd'] = np.random.normal(size=len(df))
fplt.plot(df['time'], df['rnd'], ax=ax2, color='#992277', legend='stuff')
fplt.set_y_range(ax2, -1.4, +1.7) # fix y-axis range

# finally a volume bar chart in our third plot
volume_src = fplt.PandasDataSource(df[['time','open','close','volume']])
fplt.volume_ocv(volume_src, ax=ax3)

# we're done
fplt.show()
```
