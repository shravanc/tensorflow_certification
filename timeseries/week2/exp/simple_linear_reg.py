import pandas as pd
import tensorflow as tf
from lib.utils import plot_series
import numpy as np
import matplotlib.pyplot as plt


path = "/home/shravan/python_programs/time_series/test_time_series.csv"




df = pd.read_csv(path, names=["series"])
df['series'] = df['series'] +1
series = df['series'].values
time   = list(range(0, len(series)))


split_time = 600
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
scaler = StandardScaler()
scaler = RobustScaler()
scaler = MinMaxScaler()
scaler = Normalizer()

print(x_train)
print(x_valid)
x_train = scaler.fit_transform([x_train])[0]
x_valid = scaler.fit_transform([x_valid])[0]


plot_series(time_train, x_train)
plot_series(time_valid, x_valid)
print(x_train)
print(x_valid)
# Hyper parameters
window_size = 24
batch_size = 32
shuffle_buffer_size = 500


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset




dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)

print("Layer weights {}".format(l0.get_weights()))

forecast = []

for time in range(len(series) - window_size):
  pr = series[time: time+window_size][np.newaxis]
  pr = scaler.fit_transform(pr)
  forecast.append( model.predict(pr) )
  #forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
