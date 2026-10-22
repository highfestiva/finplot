import pandas as pd

import finplot


class _ViewBox:
    def __init__(self, datasrc):
        self.datasrc = datasrc
        self.x_indexed = True


class _Axis:
    def __init__(self, datasrc):
        self.vb = _ViewBox(datasrc)


def test_short_candles_map_epoch_queries_to_positions():
    # The three candle times all sit inside the query, which used to take the
    # exact-match branch and index a RangeIndex with nanosecond timestamps.
    frame = pd.DataFrame(
        {
            "time": [1749462450000000000, 1749462450000000000, 1749462455000000000],
            "open": [1.0, 2.0, 3.0],
            "close": [1.1, 2.1, 3.1],
            "high": [1.2, 2.2, 3.2],
            "low": [0.9, 1.9, 2.9],
        }
    )
    query = pd.Series(
        [
            1749462450000000000,
            1749462451000000000,
            1749462452000000000,
            1749462453000000000,
            1749462454000000000,
            1749462455000000000,
        ]
    )

    positions = finplot._pdtime2index(_Axis(finplot.PandasDataSource(frame)), query)

    assert len(positions) == len(query)
    assert min(positions) >= 0
    assert max(positions) <= len(frame) - 1
    assert positions[-1] == len(frame) - 1
