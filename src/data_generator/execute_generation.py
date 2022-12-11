from generator import Generator
import os

COMPANIES = ['AAPL']
FROM_DATE = '2018-01-01'
TO_DATE = '2021-01-01'

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_results_path = os.path.join(root_path, 'results', 'data')


if __name__ == '__main__':
    generator = Generator(data_results_path, COMPANIES, FROM_DATE, TO_DATE)
    generator.generate()
