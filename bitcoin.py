import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

BIT_COIN_CSV_URL = 'http://www.quandl.com/api/v1/datasets/BCHARTS/KRAKENEUR.csv?api_key=xBLitG1foBj4M2YBXPxy'
df = pd.read_csv(BIT_COIN_CSV_URL, header=0,
                  index_col='Date',
                  parse_dates=True)
df.head()
df = df.iloc[::-1]
print(df.head())
features = df[["Open", "High", "Low", "Close"]].values
price_variation = (1 - (features[:, 0]/features[:, 3]))*100
highs = (features[:, 1]/np.maximum(features[:, 0], features[:,3]) - 1) * 100
lows = (features[:, 2]/np.minimum(features[:, 0], features[:, 3]) - 1) * 100
X = np.array([price_variation, highs, lows]).transpose()
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(X[0])
Y = np.array((np.sign((features[2:,3]/features[:-2,3]-1))+1)/2)
print(Y[:5])
X.shape
Y.shape
X_train = X[:1115]
X_test = X[1115:1590]
Y_train = Y[:1115]
Y_test = Y[1115:1590]
model = Sequential()
model.add(LSTM(100,
               input_shape=(None, 1),
               return_sequences=True
              ))
model.add(Dropout(0.1))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="mse", optimizer="rmsprop")
model.fit(X_train,Y_train, batch_size=512, epochs=500, validation_split=0.05)
model.evaluate(X_test,Y_test)
