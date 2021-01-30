
# https://www.kaggle.com/fatmakursun/time-series-forecasting-unknown-future
# https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816


from yahoo_fin.stock_info import get_data
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

list = ["AKBNK.IS" , "ULKER.IS" , "DGKLB.IS"]

for ticker in list:
    # ticker = "AKBNK.IS"

    def get_price():
        csv = get_data(ticker, start_date=None, end_date=None, index_as_date=False, interval="1d")
        csv.to_csv(ticker + '.csv', index=True)
        dataset = pd.read_csv(ticker + '.csv', index_col=0)
        df_stock = dataset.copy()
        df_stock = df_stock.dropna()
        return df_stock


    df = get_price()
    df = df.drop(columns=["ticker"])
    print(df.info())

    close_data = df['close'].values
    close_data = close_data.reshape((-1, 1))
    split_percent = 0.90
    split = int(split_percent*len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = df['date'][:split]
    date_test = df['date'][split:]

    print(len(close_train))
    print(len(close_test))

    look_back = 1

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=32)
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=32)

    num_epochs = 300
    batch_size = 32
    learning_rate = 0.0001

    # Initialising the LSTM
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_generator, epochs=300, verbose=1)

    prediction = model.predict(test_generator)
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))
    close_data = close_data.reshape((-1))


    def predict(num_prediction, model):
        prediction_list = close_data[-look_back:]
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
        return prediction_list


    def predict_dates(num_prediction):
        last_date = df['date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates


    num_prediction = 7
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    trace1 = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines',
        name='Train Data'
    )
    trace2 = go.Scatter(
        x=date_test,
        y=prediction,
        mode='lines',
        name='Prediction Value'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=close_test,
        mode='lines',
        name='Real Close Test Value'
    )

    trace4 = go.Scatter(
        x=df['date'].tolist(),
        y=close_data,
        mode='lines',
        name='Real Close Data'
    )
    trace5 = go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        name='Future'
    )
    layout = go.Layout(
        title=ticker + "Stock",
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )

    fig2 = go.Figure(data=[trace1, trace2, trace3, trace5], layout=layout)
    # fig2.show()
    fig2.write_html(ticker + '3.html', auto_open=False)
