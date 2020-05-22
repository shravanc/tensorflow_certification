import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from lib.utils import plot_series, trend, seasonality, noise

time = np.arange(4 * 365 + 1, dtype="float32")

baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)
plot_series(time, series)

split_time = 1000

x_train = series[:split_time]
x_valid = series[split_time:]

time_train = time[:split_time]
time_valid = time[split_time:]

ws = 20
bs = 20
sb = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


train_set = windowed_dataset(x_train, ws, bs, sb)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                           strides=1, padding='causal',
                           activation='relu',
                           input_shape=[None, 1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 200.)
])

# lambda epoch: 1e-8 * 10**(epoch / 20)
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8*10**(epoch/20)
# )

model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
    metrics=['mae']
)

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
