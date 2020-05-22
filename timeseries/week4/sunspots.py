import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lib.utils import plot_series

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

url = './Sunspots.csv'
df = pd.read_csv(url)
print(df.head())

series = df['Monthly Mean Total Sunspot Number'].values
time   = df['Unnamed: 0'].values

series = np.array(series)
time   = np.array(time)

split_time = 1000
x_train = series[:split_time]
x_valid = series[split_time:]

time_train = time[:split_time]
time_valid = time[split_time:]

plt.figure(figsize=(20,10))
plot_series(time, series)
plt.show()

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

ws = 64
bs = 256
sb = 3000
train_set = windowed_dataset(x_train, ws, bs, sb)

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                           strides=1, padding='causal',
                           activation='relu',
                           input_shape=[None, 1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(60, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(60, return_sequences=True)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*400.)
])

model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
    metrics=['mae']
)

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8*10**(epoch/20)
# )
history = model.fit(
    train_set,
    epochs=100,
    # callbacks=[lr_schedule]
)

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 30])
# plt.show()

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


rnn_forecast = model_forecast(model, series[..., np.newaxis], ws)
rnn_forecast = rnn_forecast[split_time - ws:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()

print(tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())


