significant_decimals = 8
significant_eps = 1e-8
max_zoom_points = 20 # number of visible candles when maximum zoomed in
time_splits = [('years', 2*365*24*60*60,  'YS',  4), ('months', 3*30*24*60*60, 'MS', 10), ('weeks',   3*7*24*60*60, 'W-MON', 10),
               ('days',      3*24*60*60,   'D', 10), ('hours',        9*60*60, '3H', 16), ('hours',        3*60*60,     'H', 16),
               ('minutes',        45*60, '15T', 16), ('minutes',        15*60, '5T', 16), ('minutes',         3*60,     'T', 16),
               ('seconds',           45, '15S', 19), ('seconds',           15, '5S', 19), ('seconds',            3,     'S', 19),
               ('milliseconds',       0,   'L', 23)]
