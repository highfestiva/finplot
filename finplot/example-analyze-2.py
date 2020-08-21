#!/usr/bin/env python3

import finplot as fplt
import yfinance as yf


btc = yf.download('BTC-USD', '2014-09-01', '2020-08-21')

ax1,ax2,ax3,ax4 = fplt.create_plot('Bitcoin/Dollar analysis', rows=4, maximize=False)

fplt.plot(btc.Close, color='#000', legend='Price', ax=ax1)
fplt.plot(btc.Close.rolling(200).mean(), legend='MA200', ax=ax1)
fplt.plot(btc.Close.rolling(50).mean(), legend='MA50', ax=ax1)

daily_ret = btc.Close.pct_change()*100
fplt.plot(daily_ret, width=3, color='#000', legend='Daily returns %', ax=ax2)

fplt.add_legend('Daily returns histogram', ax=ax3)
fplt.hist(daily_ret, bins=60, ax=ax3)

fplt.add_legend('Yearly returns in %', ax=ax4)
fplt.bar(btc.Close.resample('Y').last().pct_change().dropna(), ax=ax4)

fplt.show()
