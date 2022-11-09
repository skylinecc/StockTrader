import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

msft = yf.Ticker("MSFT")
hist = msft.history(period="5y", interval='1d')
df = hist.reset_index()['Close']

scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

train_size = int(len(df1) * 0.65)
val_size = int(len(df1) * 0.15)
test_size = len(df1) - train_size - val_size
train_data, val_data, test_data = df1[0:train_size, :], df1[train_size:train_size + val_size, :], \
                                  df1[train_size + val_size:len(df1), :]


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

model = tf.keras.Sequential((
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(1))
)
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=50,
    validation_data=(X_val, y_val),
)

y_preds = scaler.inverse_transform(model.predict(X_test))

rmse = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_preds))
print("RMSE: " + str(rmse))

plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)))
plt.plot(y_preds)
plt.show()

with open('testmodel.pkl', 'wb') as f:
    pickle.dump(model, f)
f.close()
