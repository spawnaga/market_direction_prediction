import sqlite3
import numpy as np
import pandas as pd
from ib_insync import *
import random
import time
import datetime


class StreamData:
    def __init__(self, db_path="./ES_ticks.db"):
        self.df = pd.DataFrame(index=range(10), columns='bidSize bidPrice askPrice askSize'.split())
        column_names = [f'{x}_{i}' for i in range(1, 11) for x in ['bidSize', 'bidPrice', 'askPrice', 'askSize']]
        column_names += ['lastPrice', 'lastSize', 'time']
        self.df_x = pd.DataFrame(index=range(1), columns=column_names)
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7496, random.randint(0, 100))
        self.contract = ContFuture(symbol="ES", exchange="CME", currency="USD")
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
                self.contract = ContFuture(symbol="ES", exchange="CME", currency="USD")
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
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} (' +
                       ', '.join([f'{name} real' for name in self.df_x.columns]) + ')')
        bids = ticker.domBids
        for i in range(10):
            self.df.iloc[i, 0] = bids[i].size if i < len(bids) else 0
            self.df.iloc[i, 1] = bids[i].price if i < len(bids) else 0

        asks = ticker.domAsks
        for i in range(10):
            self.df.iloc[i, 2] = asks[i].price if i < len(asks) else 0
            self.df.iloc[i, 3] = asks[i].size if i < len(asks) else 0

        self.last_price = ticker.domTicks[0].price if ticker.domTicks else 0
        self.last_size = ticker.domTicks[0].size if ticker.domTicks else 0

        x = self.df.values.flatten()
        x = np.append(x, [self.last_price, self.last_size, datetime.datetime.now()])

        self.df_x.iloc[0, :] = x
        self.df_x.to_sql(name=table_name, con=conn, if_exists='append', index=False)
        conn.commit()
        conn.close()


def main():
    stream = StreamData('./ES_ticks.db')
    stream.start_stream()


if __name__ == "__main__":
    main()
