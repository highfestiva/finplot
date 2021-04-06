#!/usr/bin/env python3

import finplot as fplt
import numpy as np
import pandas as pd


dates = pd.date_range('01:00', '01:00:01.200', freq='1ms')
prices = pd.Series(np.random.random(len(dates))).rolling(30).mean() + 4
fplt.plot(dates, prices, width=3)
line = fplt.add_line((dates[100], 4.4), (dates[1100], 4.6), color='#9900ff', interactive=True)
## fplt.remove_primitive(line)
text = fplt.add_text((dates[500], 4.6), "I'm here alright!", color='#bb7700')
## fplt.remove_primitive(text)
rect = fplt.add_rect((dates[700], 4.5), (dates[850], 4.4), color='#8c8', interactive=True)
## fplt.remove_primitive(rect)

def save():
    fplt.screenshot(open('screenshot.png', 'wb'))
fplt.timer_callback(save, 0.5, single_shot=True) # wait some until we're rendered

fplt.show()
