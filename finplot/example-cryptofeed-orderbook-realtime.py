#!/usr/bin/env python3
'''Slightly more advanced example which requires "pip install cryptofeed" to run. 
The data is fetched in another process and sent to the main gui thread via Pipe.
'''


from decimal import Decimal
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

import dateutil.parser
import finplot as fplt
import pandas as pd
import pytz
import requests
from cryptofeed import FeedHandler
from cryptofeed.callback import BookCallback, BookUpdateCallback, TradeCallback
from cryptofeed.defines import L2_BOOK, BOOK_DELTA, TRADES
from cryptofeed.exchanges import (Binance)
from sortedcontainers import SortedDict

utc2timestamp = lambda s: int(dateutil.parser.parse(s).replace(tzinfo=pytz.utc).timestamp() * 1000)


def download_price_history(symbol, start_time, end_time, interval_mins):
    interval_ms = 1000 * 60 * interval_mins
    interval_str = '%sm' % interval_mins if interval_mins < 60 else '%sh' % (interval_mins // 60)
    start_time = utc2timestamp(start_time)
    end_time = utc2timestamp(end_time)
    data = []
    for start_t in range(start_time, end_time, 1000 * interval_ms):
        end_t = start_t + 1000 * interval_ms
        if end_t >= end_time:
            end_t = end_time - interval_ms
        url = 'https://www.binance.com/fapi/v1/klines?interval=%s&limit=%s&symbol=%s&startTime=%s&endTime=%s' % (
            interval_str, 1000, symbol, start_t, end_t)
        print(url)
        d = requests.get(url).json()
        data += d
    df = pd.DataFrame(data, columns='t o h l c v ax bx cx dx ex fx'.split())
    return df.astype({'t': 'datetime64[ms]', 'o': float, 'h': float, 'l': float, 'c': float, 'v': float})


class RealtimePlotter:
    def __init__(self, exchange='Binance', symbol='BTCUSDT', start_time='2020-12-08', end_time='2020-12-10', interval_mins=1, num_levels = 50):
        self.num_levels = num_levels
        self.orderbook = None
        self.trade_output, self.trade_input = Pipe()
        self.book_output, self.book_input = Pipe()
        self.plots = []
        self.ohlcv = download_price_history(symbol, start_time, end_time, interval_mins)
        self.ax = fplt.create_plot(f'Realtime {exchange} {symbol}', init_zoom_periods=720, maximize=False)

    def update_candlestick_data(self, trade, interval_mins=1):
        t = int(trade['timestamp'])
        t -= t % (60 * interval_mins)
        t = pd.to_datetime(t, unit='s')
        last_price = float(trade['price'])
        amount = float(trade['amount'])
        t_last = self.ohlcv['t'].iloc[-1]
        if t < t_last:
            # ignore already-recorded trades
            return
        elif t > t_last:
            # add new candle
            o = self.ohlcv['c'].iloc[-1]
            h = last_price if last_price > o else o
            l = o if o < last_price else last_price
            df = pd.DataFrame(dict(t=[t], o=[o], c=[last_price], h=[l], l=[l], v=[amount]))
            self.ohlcv = pd.concat([self.ohlcv, df], ignore_index=True, sort=False)
        else:
            # update last candle
            i = self.ohlcv.index.max()
            self.ohlcv.loc[i, 'c'] = last_price
            self.ohlcv.loc[i, 'v'] += amount
            if last_price > self.ohlcv.loc[i, 'h']:
                self.ohlcv.loc[i, 'h'] = last_price
            if last_price < self.ohlcv.loc[i, 'l']:
                self.ohlcv.loc[i, 'l'] = last_price

    def aggregate_ordebook(self):
        bids = self.orderbook['book']['bid']
        asks = self.orderbook['book']['ask']
        bids['price'] -= 0.5
        bids['volume'] = -bids['volume'].cumsum()
        asks['volume'] = -asks['volume'].cumsum()
        aggregated_book = pd.concat([bids.iloc[::-1], asks])
        return [[len(self.ohlcv) + 0.5, aggregated_book]]

    def update_plot(self):
        while self.trade_output.poll():
            trade = self.trade_output.recv()
            self.update_candlestick_data(trade)
            # print(f'trade {trade["price"]} {trade["amount"]} {trade["timestamp"]}')

        while self.book_output.poll():
            self.orderbook = self.book_output.recv()
            print(f'top {self.orderbook["book"]["bid"].iloc[0].values.tolist()} <-> {self.orderbook["book"]["ask"].iloc[0].values.tolist()}')

        candlesticks = self.ohlcv['t o c h l'.split()]
        volumes = self.ohlcv['t o c v'.split()]

        if not self.plots:  # 1st time
            candlestick_plot = fplt.candlestick_ochl(candlesticks)
            self.plots.append(candlestick_plot)
            self.plots.append(fplt.volume_ocv(volumes, ax=self.ax.overlay()))
            x = len(candlesticks)+2.5
            y = candlesticks.c.iloc[-1]
            orderbook = [[x,[(y,1)]]]
            orderbook_colorfunc = fplt.horizvol_colorfilter([(0,'bull'),(10,'bear')])
            orderbook_plot = fplt.horiz_time_volume(orderbook, candle_width=1, draw_body=10, colorfunc=orderbook_colorfunc)
            self.plots.append(orderbook_plot)

            candlestick_plot.colors.update(dict(
                bull_shadow='#388d53',
                bull_frame='#205536',
                bull_body='#52b370',
                bear_shadow='#d56161',
                bear_frame='#5c1a10',
                bear_body='#e8704f'))
            orderbook_plot.colors.update(dict(
                bull_frame='#52b370',
                bull_body='#bae1c6',
                bear_frame='#e8704f',
                bear_body='#f6c6b9'))
        else:
            self.plots[0].update_data(candlesticks)
            self.plots[1].update_data(volumes)
            if self.orderbook is not None:
                aggregated = self.aggregate_ordebook()
                self.plots[2].update_data(aggregated)


class SingleExchangeFeedHandler:
    def __init__(self, symbol: str, num_levels: int, trade_input: Connection, book_input: Connection):
        self.symbol = symbol
        self.num_levels = num_levels
        self.trade_input = trade_input
        self.book_input = book_input
        self.orderbook = None
        self.oderbook_bid_view = None
        self.oderbook_ask_view = None

    def apply_book_delta(self, delta: dict, side: str):
        delta = delta[side]
        book: SortedDict = self.orderbook[side]
        for (level, quantity) in delta:
            if level in book:
                if quantity == Decimal('0E-8'):
                    del book[level]
                else:
                    book[level] = quantity

    async def trade(self, feed, pair, order_id, timestamp, side, amount, price, receipt_timestamp):
        assert isinstance(timestamp, float)
        assert isinstance(side, str)
        assert isinstance(amount, Decimal)
        assert isinstance(price, Decimal)
        data = dict(
            feed=feed,
            pair=pair,
            order_id=order_id,
            timestamp=timestamp,
            side=side,
            amount=amount,
            price=price,
            receipt_timestamp=receipt_timestamp
        )
        self.trade_input.send(data)
        #print(f'----> trade {price} {amount} {timestamp}')

    async def book(self, feed, pair, book, timestamp, receipt_timestamp):
        data = dict(
            feed=feed,
            pair=pair,
            book=book,
            timestamp=timestamp,
            receipt_timestamp=receipt_timestamp
        )
        self.orderbook = book
        self.oderbook_bid_view = book['bid'].items()
        self.oderbook_ask_view = book['ask'].items()

    async def book_update(self, feed, pair, delta, timestamp, receipt_timestamp):
        self.apply_book_delta(delta, 'bid')
        self.apply_book_delta(delta, 'ask')
        nb = min(self.num_levels, len(self.oderbook_bid_view))
        na = min(self.num_levels, len(self.oderbook_ask_view))
        bids = pd.DataFrame(self.oderbook_bid_view[-1:-(nb+1):-1], columns=['price', 'volume']).astype(float)
        asks = pd.DataFrame(self.oderbook_ask_view[0:na:1], columns=['price', 'volume']).astype(float)
        data = dict(
            feed=feed,
            pair=pair,
            book=dict(bid=bids, ask=asks),
            timestamp=timestamp,
            receipt_timestamp=receipt_timestamp
        )
        self.book_input.send(data)
        # print(f'----> {self.orderbook["bid"].items()[-1]} <-> {self.orderbook["ask"].items()[0]}')

    def run(self):
        f = FeedHandler()
        f.add_feed(Binance(pairs=[self.symbol],
                           channels=[TRADES, L2_BOOK],
                           callbacks={
                               TRADES: TradeCallback(self.trade),
                               L2_BOOK: BookCallback(self.book),
                               BOOK_DELTA: BookUpdateCallback(self.book_update),
                           }))

        print(f'starting crypto market feed handler')
        f.run()


if __name__ == '__main__':
    plotter = RealtimePlotter(symbol='BTCUSDT', start_time='2020-12-09', num_levels=200)

    feed_handler = SingleExchangeFeedHandler(
        symbol='BTC-USDT',
        num_levels=200,
        trade_input=plotter.trade_input,
        book_input=plotter.book_input
    )

    p = Process(target=feed_handler.run, args=())
    p.start()

    plotter.update_plot()
    fplt.timer_callback(plotter.update_plot, 0.1)
    fplt.show()
