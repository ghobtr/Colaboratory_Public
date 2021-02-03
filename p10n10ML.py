# univariate lstm example
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from yahoo_fin.stock_info import get_data
import plotly.graph_objects as go


# Global vars
# stock_list = ["ANHYT.IS","AEFES.IS","KCHOL.IS","SAHOL.IS","CCOLA.IS","VAKBN.IS","AKBNK.IS"]
stock_list = ["VESTL.IS","ANHYT.IS","ARCLK.IS","AEFES.IS","TKFEN.IS","SISE.IS","TUPRS.IS","ULKER.IS","KCHOL.IS","SAHOL.IS","VAKBN.IS","AKBNK.IS"]

for ticker in stock_list:

    split_percent = 0.99


    def get_price():
        df_stock = get_data(ticker, start_date=None, end_date=None, index_as_date=False, interval="1d")
        df_stock = df_stock.dropna()
        return df_stock

    # preparing independent and dependent features
    def prepare_data(timeseries_data, n_features):
        X, y = [], []
        for i in range(len(timeseries_data)):
            # find the end of this pattern
            end_ix = i + n_features
            # check if we are beyond the sequence
            if end_ix > len(timeseries_data) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


    df = get_price()
    close_data = df['close']
    # define input sequence

    n_steps = 10
    split = int(split_percent * len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]
    close_train = np.array(close_train)
    close_test = np.array(close_test)

    # split into samples
    X_train, y_train = prepare_data(close_train, n_steps)
    X_test, y_test = prepare_data(close_test, n_steps)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # model = Sequential([
    #     LSTM(120,input_shape=(n_steps, n_features),return_sequences=True),
    # #     Dropout(0.3),
    #     LSTM(80,return_sequences=True),
    # #     Dropout(0.3),
    #     LSTM(60,return_sequences=False),
    # #     Dense(20),
    #     Dense(1)
    # ])
    # model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

    # fit model
    print("Model Train Start")
    model.fit(X_train, y_train, epochs=300, verbose=1)
    print("Model Train Finish")

    print("Model Test Prediction Start")
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_predict_test = model.predict(X_test, verbose=1)

    y_predict_test = y_predict_test.reshape(-1)
    print("Model Test Prediction Finish")

    # Predicting For the next 10 data

    # demonstrate prediction for next 10 days
    x_input = y_test[-10:]
    temp_input = list(x_input)
    lst_output = []
    i = 0
    while (i < 10):

        if (len(temp_input) > 10):
            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            # print(x_input)
            x_input = x_input.reshape((1, n_steps, n_features))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.append(yhat[0][0])
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i = i + 1

    print(lst_output)

    # future_days = shift(lst_output, len(y_test), cval=np.NaN)

    # show plt
    trace1 = go.Scatter(
        # x=train_data.index.tolist(),
        y=y_test,
        mode='lines',
        name='Real Data'
    )
    trace2 = go.Scatter(
        # x=test_data.index.tolist(),
        y=y_predict_test,
        mode='lines',
        name='Predict Data'
    )
    trace3 = go.Scatter(
        # x=test_data.index.tolist(),
        y=lst_output,
        mode='lines',
        name='Future Data'
    )

    layout = go.Layout(
        title=ticker + "Stock",
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )

    fig2 = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig2.write_html(ticker + 'p10n10.html', auto_open=False)
