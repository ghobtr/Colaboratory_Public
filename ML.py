# Import modules and packages
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from yahoo_fin.stock_info import get_data
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
# Import Libraries and packages from Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam


def get_price(ticker):
    csv = get_data(ticker, start_date=None, end_date=None, index_as_date=False, interval="1d")
    csv = csv.drop(columns=['ticker', 'adjclose'])
    csv.to_csv(ticker + '.csv', index=False, header=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])


ticker = "GARAN.IS"
get_price(ticker)

# Importing Training Set
dataset_train = pd.read_csv(ticker + '.csv')
# dataset_train = pd.read_csv("AKBNK.IS.csv")
print(dataset_train.info())
# Select features (columns) to be involved intro training and predictions
cols = list(dataset_train)[1:5]

# Extract dates (will be used in visualization)
datelist_train = list(dataset_train['Date'])
datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]

print('Training set shape == {}'.format(dataset_train.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print('Featured selected: {}'.format(cols))

dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)

# Using multiple features (predictors)
training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))


# Feature Scaling

sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])

# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 60   # Number of days we want top predict into the future
n_past = 90     # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))

# Initializing the Neural Network based on LSTM
model = Sequential()

# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))

# Adding 2nd LSTM layer
model.add(LSTM(units=10, return_sequences=False))

# Adding Dropout
model.add(Dropout(0.25))

# Output layer
model.add(Dense(units=1, activation='linear'))

# Compiling the Neural Network
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')


es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

tb = TensorBoard('logs')

history = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)

# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()
print(datelist_future)

'''
Remeber, we have datelist_train from begining.
'''

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

# Perform predictions
predictions_future = model.predict(X_train[-n_future:])

predictions_train = model.predict(X_train[n_past:])

# Inverse the predictions to original measurements

# ---> Special function: convert <datetime.date> to <Timestamp>


def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)

    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(pd.Series(datelist_train[2 * n_past + n_future - 1:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

# print(PREDICTION_TRAIN.head(3))
# print(PREDICTIONS_FUTURE.head(3))
#
# print(PREDICTION_TRAIN.info())
# print(PREDICTIONS_FUTURE.info())

# # Set plot size
# from pylab import rcParams
# rcParams['figure.figsize'] = 14, 5
#
# # Plot parameters
# START_DATE_FOR_PLOTTING = '2020-12-22'
#
# plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Open'], color='r', label='Predicted Stock Price')
# plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
# plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')
#
# plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')
#
# plt.grid(which='major', color='#cccccc', alpha=0.5)
#
# plt.legend(shadow=True)
# plt.title('Predcitions and Acutal Stock Prices', family='Arial', fontsize=12)
# plt.xlabel('Timeline', family='Arial', fontsize=10)
# plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
# plt.xticks(rotation=45, fontsize=8)
# plt.show()

#
print(dataset_train.index.values)
trace1 = go.Scatter(
    x = PREDICTIONS_FUTURE.index,
    y = PREDICTIONS_FUTURE['Open'],
    mode='lines',
    name='Prediction Data'
 )
trace2 = go.Scatter(
    x=PREDICTION_TRAIN.index,
    y=PREDICTION_TRAIN['Open'],
    mode='lines',
    name='Predict Train Value'
)
trace3 = go.Scatter(
    x=datelist_train,
    y=dataset_train['Open'],
    mode='lines',
    name='Real Close  Value'
)
#
# trace4 = go.Scatter(
#     x=df['date'].tolist(),
#     y=close_data,
#     mode='lines',
#     name='Real Close Data'
# )
# trace5 = go.Scatter(
#     x=forecast_dates,
#     y=forecast,
#     mode='lines',
#     name='Future'
# )
layout = go.Layout(
    title=ticker + "Stock",
    xaxis={'title': "Date"},
    yaxis={'title': "Close"}
)
#
fig2 = go.Figure(data=[trace1, trace2, trace3], layout=layout)
# fig2 = go.Figure(data=[trace1, trace2, trace3, trace5], layout=layout)
fig2.show()
# fig2.write_html(ticker + '3.html', auto_open=False)
