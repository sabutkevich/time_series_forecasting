from model.yfinance import Yfinance
import logging
import os

logger = logging.getLogger()


class Generator:

    def __init__(self, data_results_path, companies, from_date, to_date):
        self.data_results_path = data_results_path
        self.companies = companies
        self.from_date = from_date
        self.to_date = to_date
        self.yfinance_source = Yfinance()
        self.yfinance_interval = 7

    def yfinance_generate(self, company):
        data_frame = self.yfinance_source.get_stocks(company, self.from_date, self.to_date,
                                                     Yfinance.PERIODS[self.yfinance_interval - 1],
                                                     Yfinance.INTERVALS[self.yfinance_interval - 1])
        file_name = os.path.join(self.data_results_path, 'data_yfinance.csv')
        data_frame.to_csv(file_name, header=True, sep=' ', mode='a')

    def generate(self):
        for company in self.companies:
            self.yfinance_generate(company)
