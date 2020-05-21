import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


from lib.utils import plot_series, trend, seasonal_pattern, seasonality, noise

time = np.arange(4 * 365 + 1, dtype="float32")

baseline    = 10
amplitude   = 40
slope       = 0.05
noise_level = 5


series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)


split_time  = 1000
time_train  = time[:split_time]
x_train     = series[:split_time]
time_valid  = time[split_time:]
x_valid     = series[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
#plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)



# Naive Forecast:

naive_forecast = series[split_time - 1: -1]
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)


plt.show()

# Metrics:
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

