#!/usr/bin/env python3

import finplot as fplt
import numpy as np
import pandas as pd
import yfinance as yf


btc = yf.download('BTC-USD', '2014-09-01')

ax1,ax2,ax3,ax4,ax5 = fplt.create_plot('Bitcoin/Dollar long term analysis', rows=5, maximize=False)
fplt.set_y_scale(ax=ax1, yscale='log')

fplt.plot(btc.Close, color='#000', legend='Log price', ax=ax1)
btc['ma200'] = btc.Close.rolling(200).mean()
btc['ma50'] = btc.Close.rolling(50).mean()
fplt.plot(btc.ma200, legend='MA200', ax=ax1)
fplt.plot(btc.ma50, legend='MA50', ax=ax1)
btc['one'] = 1
fplt.volume_ocv(btc[['ma200','ma50','one']], candle_width=1, ax=ax1.overlay(scale=0.02))

daily_ret = btc.Close.pct_change()*100
fplt.plot(daily_ret, width=3, color='#000', legend='Daily returns %', ax=ax2)

fplt.add_legend('Daily % returns histogram', ax=ax3)
fplt.hist(daily_ret, bins=60, ax=ax3)

fplt.add_legend('Yearly returns in %', ax=ax4)
fplt.bar(btc.Close.resample('Y').last().pct_change().dropna()*100, ax=ax4)

# calculate monthly returns, display as a 4x3 heatmap
months = btc['Adj Close'].resample('M').last().pct_change().dropna().to_frame() * 100
months.index = mnames = months.index.month_name().to_list()
mnames = mnames[mnames.index('January'):][:12]
mrets = [months.loc[mname].mean()[0] for mname in mnames]
hmap = pd.DataFrame(columns=[2,1,0], data=np.array(mrets).reshape((3,4)).T)
hmap = hmap.reset_index() # use the range index as X-coordinates (if no DateTimeIndex is found, the first column is used as X)
colmap = fplt.ColorMap([0.3, 0.5, 0.7], [[255, 110, 90], [255, 247, 0], [60, 255, 50]]) # traffic light
fplt.heatmap(hmap, rect_size=1, colmap=colmap, colcurve=lambda x: x, ax=ax5)
for j,mrow in enumerate(np.array(mnames).reshape((3,4))):
    for i,month in enumerate(mrow):
        s = month+' %+.2f%%'%hmap.loc[i,2-j]
        fplt.add_text((i, 2.5-j), s, anchor=(0.5,0.5), ax=ax5)
ax5.set_visible(crosshair=False, xaxis=False, yaxis=False) # hide junk for a more pleasing look

fplt.show()
