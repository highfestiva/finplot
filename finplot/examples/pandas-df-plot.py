#!/usr/bin/env python3

import finplot as fplt
import pandas as pd

labels = ['A', 'B', '<span style="text-decoration:underline; color:#00b">C</span>', '<span style="font-weight:bold">D</span>']
df = pd.DataFrame({'A':[1,2,3,5], 'B':[2,4,5,3], 'C':[3,6,11,8], 'D':[1,1,2,2], 'labels':labels}, index=[1606086000, 1606086001, 1606086002, 1606086003])

df.plot('A')
df.plot('A', 'labels', kind='labels', color='#b90')
df.plot('C', kind='scatter')
(1-df.B).plot(kind='scatter')
df.plot(kind='candle').setZValue(-100)
(df.B-4).plot.bar()

fplt.show()
