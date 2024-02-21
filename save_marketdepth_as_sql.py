import sqlite3
import numpy as np
import pandas as pd
from ib_insync import *
import random
import time


class StreamData:
    def __init__(self, db_path="./CL_ticks.db"):
        self.ticker_size = [0, 0]
        self.ticker_price = [0, 0]
        self.df = pd.DataFrame(index=range(5), columns='bidSize bidPrice askPrice askSize'.split())
        self.df_x = pd.DataFrame(index=range(1), columns='bidSize_1 bidPrice_1 askPrice_1 askSize_1 \
                                        bidSize_2 bidPrice_2 askPrice_2 askSize_2 bidSize_3 \
                                            bidPrice_3 askPrice_3 askSize_3 bidSize_4 \
                                                bidPrice_4 askPrice_4 askSize_4 bidSize_5 \
                                                    bidPrice_5 askPrice_5 askSize_5 lastPrice lastSize'.split())
        self.last_price = None
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7496, random.randint(0, 100))
        self.contract = ContFuture(symbol="NQ", exchange="CME", currency="USD")
        self.ib.qualifyContracts(self.contract)
        self.ticker = self.ib.reqMktDepth(self.contract)
        self.db_file = db_path
        self.db = sqlite3.connect(self.db_file)
        self.cursor = self.db.cursor()
        self.ib.errorEvent += self.errorAndReconnect  # subscribe to error event

    def errorAndReconnect(self, reqId=None, errorCode=None, errorString=None):
        print(f"Error: {errorCode}, {errorString}")
        # handle the error here as appropriate for your use case
        while not self.ib.isConnected() or not self.ib.client.isConnected():
            try:
                self.ib.connect('127.0.0.1', 7496, random.randint(0, 100))
                self.contract = ContFuture(symbol="NQ", exchange="CME", currency="USD")
                self.ib.qualifyContracts(self.contract)
                self.ticker = self.ib.reqMktDepth(self.contract)
                print("Connected to Interactive Brokers TWS")
            except Exception as e:
                print(f"Error: {e}. Retrying in 10 seconds...")
                time.sleep(10)

    def start_stream(self):
        self.ticker.updateEvent += self.on_ticker_update
        self.ib.run()

    def on_ticker_update(self, ticker):
        table_name = f'{self.contract.symbol}_market_depth'
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} '
                       '(bidSize_1 real, bidPrice_1 real, '
                       'askPrice_1 real, askSize_1 real, '
                       'bidSize_2 real, bidPrice_2 real, '
                       'askPrice_2 real, askSize_2 real, '
                       'bidSize_3 real, bidPrice_3 real, '
                       'askPrice_3 real, askSize_3 real, '
                       'bidSize_4 real, bidPrice_4 real, '
                       'askPrice_4 real, askSize_4 real, '
                       'bidSize_5 real, bidPrice_5 real, '
                       'askPrice_5 real, askSize_5 real, '
                       'lastPrice real, lastSize real)')
        bids = ticker.domBids
        for i in range(5):
            self.df.iloc[i, 0] = bids[i].size if i < len(bids) else 0
            self.df.iloc[i, 1] = bids[i].price if i < len(bids) else 0

        asks = ticker.domAsks
        for i in range(5):
            self.df.iloc[i, 2] = asks[i].price if i < len(asks) else 0
            self.df.iloc[i, 3] = asks[i].size if i < len(asks) else 0
        self.last_price = ticker.domTicks[0].price
        x = self.df.values.flatten()
        x = np.append(x, self.last_price)
        self.df_x.iloc[0, :] = x
        self.df_x.to_sql(name=table_name, con=conn, if_exists='append', index=False)
        conn.commit()
        conn.close()


def main():
    stream = StreamData('./NQ_ticks.db')
    stream.start_stream()


if __name__ == "__main__":
    main()
