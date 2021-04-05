#!/usr/bin/env python3

from datetime import date
import finplot as fplt
import requests
import pandas as pd

now = date.today().strftime('%Y-%m-%d')
r = requests.get('https://www.bitstamp.net/api-internal/tradeview/price-history/BTC/USD/?step=86400&start_datetime=2011-08-18T00:00:00.000Z&end_datetime=%sT00:00:00.000Z' % now)
df = pd.DataFrame(r.json()['data']).astype({'timestamp':int, 'open':float, 'close':float, 'high':float, 'low':float}).set_index('timestamp')

# plot price
fplt.create_plot('Bitcoin 2011-%s'%now.split('-')[0], yscale='log')
fplt.candlestick_ochl(df['open close high low'.split()])

# monthly separator lines
months = pd.to_datetime(df.index, unit='s').strftime('%m')
last_month = ''
for x,(month,price) in enumerate(zip(months, df.close)):
    if month != last_month:
        fplt.add_line((x-0.5, price*0.5), (x-0.5, price*2), color='#bbb', style='--')
    last_month = month

fplt.show()
