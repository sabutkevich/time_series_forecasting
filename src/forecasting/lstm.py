import math
import pandas as pd
import numpy as np
import os
import plotly.graph_objs as go
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger()


def predict(model):
    global adj_data
    prediction_list = adj_data[-LOOK_BACK:]
    for _ in range(PREDICTION_WINDOW):
        x = prediction_list[-LOOK_BACK:]
        x = x.reshape((1, LOOK_BACK, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[LOOK_BACK - 1:]
    return prediction_list


PREDICTION_WINDOW = 30
SPLIT_TRAIN_TEST = 90
LOOK_BACK = 15

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_results_path = os.path.join(root_path, 'results', 'data')
df_source = pd.read_csv(os.path.join(data_results_path, 'data_yfinance.csv'), sep=' ')
scaler = MinMaxScaler(feature_range=(0, 1))
df_source['Date'] = pd.to_datetime(df_source['Date'])
adj_data_source = scaler.fit_transform(df_source['AdjClose'].values.reshape((-1, 1)))

if PREDICTION_WINDOW >= len(df_source):
    exit(0)

df = df_source[:-PREDICTION_WINDOW]

df.set_axis(df['Date'], inplace=True)
adj_data = df['AdjClose'].values
adj_data = scaler.fit_transform(adj_data.reshape((-1, 1)))

split_percent = SPLIT_TRAIN_TEST * 10**-2
split = int(split_percent * len(adj_data))

adj_train = adj_data[:split + 1]
adj_test = adj_data[split:]

date_train = df['Date'][:split + 1]
date_test = df['Date'][split:]

train_generator = TimeseriesGenerator(adj_train, adj_train, length=LOOK_BACK, batch_size=20)
test_generator = TimeseriesGenerator(adj_test, adj_test, length=LOOK_BACK, batch_size=1)

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(LOOK_BACK, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 200
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
prediction = model.predict_generator(test_generator)

date_prediction = date_test.tolist()[LOOK_BACK:]

adj_train = scaler.inverse_transform(adj_train).reshape((-1))
adj_test = scaler.inverse_transform(adj_test).reshape((-1))
prediction = scaler.inverse_transform(prediction).reshape((-1))

trace1 = go.Scatter(x=date_train, y=adj_train, mode='lines', name='Обучающие данные')
trace2 = go.Scatter(x=date_prediction, y=prediction, mode='lines', name='Прогноз')
trace3 = go.Scatter(x=date_test, y=adj_test, mode='lines', name='Тестовые данные')
layout = go.Layout(title="Акции IBM", xaxis={'title': "Дата"}, yaxis={'title': "Средняя цена"})

adj_data = adj_data.reshape((-1))

forecast = predict(model)

forecast = scaler.inverse_transform(forecast.reshape((-1, 1))).reshape((-1))

rmse_prediction = math.sqrt(mean_squared_error(adj_test[LOOK_BACK:], prediction))
logger.info(f'RMSE test data: {rmse_prediction}')

rmse_forecast = math.sqrt(mean_squared_error(df_source[-PREDICTION_WINDOW - 1:]['AdjClose'], forecast))
logger.info(f'RMSE forecast: {rmse_forecast}')

trace4 = go.Scatter(x=df_source[-PREDICTION_WINDOW - 1:]['Date'],
                    y=forecast, mode='lines',
                    name='Прогноз')
trace5 = go.Scatter(x=df_source[-PREDICTION_WINDOW - 1:]['Date'],
                    y=df_source[-PREDICTION_WINDOW - 1:]['AdjClose'], mode='lines', name='Реальные данные')

fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5], layout=layout)
fig.show()
