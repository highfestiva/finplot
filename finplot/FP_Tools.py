import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


def _makepen(color, style=None, width=1):
    if style is None or style == '-':
        return pg.mkPen(color=color, width=width)
    dash = []
    for ch in style:
        if ch == '-':
            dash += [4,2]
        elif ch == '_':
            dash += [10,2]
        elif ch == '.':
            dash += [1,2]
        elif ch == ' ':
            if dash:
                dash[-1] += 2
    return pg.mkPen(color=color, style=QtCore.Qt.CustomDashLine, dash=dash, width=width)
