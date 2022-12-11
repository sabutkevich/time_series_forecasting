import yfinance


class Yfinance:
    INTERVALS = ['1m', '2m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    PERIODS = ['7d', '60d', '60d', '60d', '60d', '730d', 'max', 'max', 'max', 'max', 'max']
    COLUMN_LABELS = {'Open': 'Open',
                     'High': 'High',
                     'Low': 'Low',
                     'Close': 'Close',
                     'Volume': 'Volume',
                     'Adj Close': 'AdjClose',
                     'company': 'Company',
                     'Date': 'Date'}

    def get_stocks(self, company_name, from_date, to_date, period, interval):
        data_frame = yfinance.download(company_name, start=from_date, end=to_date, period=period,
                                       interval=interval, progress=False)
        data_frame.rename(columns=self.COLUMN_LABELS, inplace=True)
        return data_frame
