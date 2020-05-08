#!/usr/bin/env python3

import finplot as fplt
import numpy as np
import pandas as pd


dates = pd.date_range('01:00', '01:00:01.200', freq='1ms')
prices = pd.Series(np.random.random(len(dates))).rolling(30).mean() + 4
fplt.plot(dates, prices)
line = fplt.add_line((dates[100], 4.4), (dates[1100], 4.6), color='#9900ff', interactive=True)
## fplt.remove_line(line)
text = fplt.add_text((dates[500], 4.6), "I'm here alright!", color='#bb7700')
## fplt.remove_text(text)
fplt.show()
