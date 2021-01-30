
import tensorflow as tf
from tensorflow import keras
import os,inspect
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation,concatenate,LeakyReLU,SimpleRNN,GRU
from keras import optimizers
from keras.models import Sequential


# Global vars
directory_path=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


# define model


units= 50
second_units=30
output_size=16



model = Sequential()

model.add(GRU(256, return_sequences=True, input_shape=(60,5), recurrent_dropout=0.5))
model.add(SimpleRNN(10))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])



model.summary()

# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('model_p1n10ML_2.json', 'w') as json_file:
    json_file.write(json_model)
