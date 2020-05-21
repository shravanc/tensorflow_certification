import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from lib.utils import plot_series, trend, seasonal_pattern, seasonality, noise


#=======================DataCreation=====================
time = np.arange(4 * 465 + 1, dtype="float32")

baseline    = 10
amplitude   = 40
slope       = 0.05
noise_level = 5

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

#=======================DataCreation=====================


#=======================DataPreparation==================
split_time = 1000

x_train     = series[:split_time]
time_train  = time[:split_time]

x_valid     = series[split_time:]
time_valid  = time[split_time:]

window_size = 20
batch_size  = 32
shuffle_buffer = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
  dataset = dataset.map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.shuffle(shuffle_buffer)
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer)
#=======================DataPreparation==================


#=======================MODEL=============================
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=[window_size]),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])


model.compile(
  loss='mse',
  optimizer=tf.keras.optimizers.SGD( lr=1e-6, momentum=0.9 )
)


history = model.fit(
  dataset,
  epochs=100
)
#=======================MODEL=============================

#=======================ForeCast==========================

forecast = []
for time in range(len(series)-window_size):
  input_data = series[time:time+window_size][np.newaxis]
  prediction = model.predict(input_data)
  forecast.append(prediction)


forecast = forecast[split_time-window_size:]
results  = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(20, 10))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

#=======================ForeCast==========================

#=======================Metrics===========================

score = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
print(f"MAE score: {score}")

#=======================Metrics===========================

