import pandas as pd
import quandl
import math,datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
print(df)
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))

X = X[:- forecast_out]
X_lately =X [- forecast_out:]
X = preprocessing.scale(X)
df.dropna(inplace=True)

y = np.array(df['label'])
#y = np.array(df['label'])
X_train, X_test, y_train , y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf=LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
acuracy=clf.score(X_test,y_test)
#print(acuracy)
forecast_set=clf.predict(X_lately)
print(forecast_set, acuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix =last_unix + one_day
