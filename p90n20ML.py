# univariate lstm example
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, GRU, Reshape, concatenate
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from yahoo_fin.stock_info import get_data
import plotly.graph_objects as go
import time
from keras.models import Model
from keras import optimizers
from keras.models import load_model,model_from_json
import json
import os, inspect
from sklearn import metrics


# Global vars

stock_list = ["VESTL.IS","ANHYT.IS","ARCLK.IS","AEFES.IS","TKFEN.IS","SISE.IS","TUPRS.IS","ULKER.IS","KCHOL.IS","SAHOL.IS","AKBNK.IS","VAKBN.IS","THYAO.IS"]
stock_list = ["AKBNK.IS","THYAO.IS"]
epochs_number = 100
batch_size_number = 32
directory_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
modeljson = directory_path+'/model_p90n20ML_1.json'

for ticker in stock_list:

    split_percent = 0.95

    def get_price():
        df_stock = get_data(ticker, start_date=None, end_date=None, index_as_date=False, interval="1d")
        #df_stock = get_data(ticker, start_date=None, end_date='01/22/2021', index_as_date=False, interval="1d")
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

    n_steps = 90
    split = int(split_percent * len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]
    close_train = np.array(close_train)
    close_test = np.array(close_test)

    # split into samples
    X_train, y_train = prepare_data(close_train, n_steps)
    X_test, y_test = prepare_data(close_test, n_steps)



    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))

    # define model

    dataj = json.load(open(modeljson))
    mymodel = json.dumps(dataj)

    model = model_from_json(mymodel)
    # model = load_model('my_model.h5')
    # model.compile(loss=mean_absolute_error, optimizer=rmsprop)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit model
    print("Model Train Start")
    # model.fit(x=X_train, y=y_train, batch_size=32, epochs=350, shuffle=True, validation_split=0.1)
    # model.fit(x=X_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    model.fit(x = X_train, y = y_train, epochs = epochs_number, batch_size = batch_size_number)
    # model.fit(x = X_train, y = y_train, epochs = 50, batch_size = 32)
    print("Model Train Finish")

    print("Model Test Prediction Start")
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_predict_test = model.predict(X_test, verbose=0)

    y_predict_test = y_predict_test.reshape(-1)
    print("Model Test Prediction Finish")

    # Predicting For the next 10 data

    # demonstrate prediction for next 10 days
    x_input = y_test[-90:]
    temp_input = list(x_input)
    lst_output = []
    i = 0
    while (i < 20):

        if (len(temp_input) > 90):
            x_input = np.array(temp_input[1:])
            # print("{} day input {}".format(i, x_input))
            # print(x_input)
            x_input = x_input.reshape((1, n_steps, n_features))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i, yhat))
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

    

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict_test))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_test))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_test)))

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
    fig2.write_html(ticker + '_p90n20ML_model1.html', auto_open=False)
