import math
import pandas as pd
import pmdarima as pm
import os
import plotly.graph_objs as go
from numpy import log
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ARIMA
import logging

from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger()


def get_model(data):
    return pm.auto_arima(data, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3,
                         m=1,
                         d=None,
                         seasonal=False,
                         start_P=0,
                         D=0,
                         trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)


PREDICTION_WINDOW = 30
SPLIT_TRAIN_TEST = 90

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_results_path = os.path.join(root_path, 'results', 'data')
df_source = pd.read_csv(os.path.join(data_results_path, 'data_yfinance.csv'), sep=' ')
df_source['Date'] = pd.to_datetime(df_source['Date'])
df = df_source[:-PREDICTION_WINDOW]
adj_data = df['AdjClose'].values

adj_data_log = log(adj_data)
result = adfuller(adj_data_log)
print('Коэффициент расширенного теста Дики-Фуллера: %f' % result[0])
print('Критические значения: %f' % result[1])
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

split_percent = SPLIT_TRAIN_TEST * 10 ** -2
split = int(split_percent * len(adj_data))

adj_train = adj_data[:split + 1]
adj_test = adj_data[split:]

date_train = df['Date'][:split + 1]
date_test = df['Date'][split:]

auto_model = get_model(adj_data)

history = [x for x in adj_train]
predictions = []

for t in range(len(adj_test)):
    model = ARIMA(history, order=auto_model.order)
    model_fit = model.fit()
    predictions.append(model_fit.forecast()[0])
    history.append(adj_test[t])

pc_series = pd.Series(predictions, index=date_test)

fc = auto_model.predict(n_periods=PREDICTION_WINDOW)

rmse_prediction = math.sqrt(mean_squared_error(adj_test, predictions))
logger.info(f'RMSE test data: {rmse_prediction}')

rmse_forecast = math.sqrt(mean_squared_error(df_source[-PREDICTION_WINDOW:]['AdjClose'], fc))
logger.info(f'RMSE forecast: {rmse_forecast}')

trace1 = go.Scatter(x=date_train, y=adj_train, mode='lines', name='Обучающие данные')
trace2 = go.Scatter(x=df_source['Date'][-PREDICTION_WINDOW - 1:], y=fc, mode='lines',
                    name='Прогноз')
trace3 = go.Scatter(x=date_test, y=pc_series, mode='lines', name='Прогноз')
trace4 = go.Scatter(x=date_test, y=adj_test, mode='lines', name='Тестовые данные')
trace5 = go.Scatter(x=df_source['Date'][-PREDICTION_WINDOW - 1:],
                    y=df_source['AdjClose'][-PREDICTION_WINDOW - 1:], mode='lines',
                    name='Реальные данные')
layout = go.Layout(title="Акции IBM", xaxis={'title': "Дата"}, yaxis={'title': "Средняя цена"})
fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5], layout=layout)
fig.show()
