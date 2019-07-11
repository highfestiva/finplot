#!/usr/bin/env python3

import finplot as fplt
import numpy as np
import pandas as pd


dates = pd.date_range('01:00', '21:30', freq='1min')
prices = pd.Series(np.random.random(len(dates))).rolling(30).mean() + 4
plot = fplt.plot(dates, prices)
line = fplt.add_line(plot.ax, (dates[100].timestamp(), 4.4), (dates[1100].timestamp(), 4.6), color='#9900ff', interactive=True)
#fplt.remove_line(plot.ax, line)
text = fplt.add_text(plot.ax, (dates[500].timestamp(), 4.6), "I'm here alright!", color='#bb7700')
#fplt.remove_text(plot.ax, text)
fplt.show()
