import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/nov_time_series.csv'
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/one_year_timeseries.csv'

df = pd.read_csv(url)
print(df.describe())
df['n_series'] = df['series'] + 80

raw_df = df.copy()

train_df = df.sample(frac=0.99)
valid_df = df.drop(train_df.index)

train_time = range(len(train_df))
valid_time = range(len(valid_df))


def norm(data):
    data['n_series'] = stats.boxcox(data['n_series'])[0]


norm(train_df)
norm(valid_df)
norm(raw_df)

train_data = train_df['n_series'].values
valid_data = valid_df['n_series'].values
series = raw_df['n_series'].values

ws = 24
bs = 12
sb = 100


def windowed_dataset(t_series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(t_series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


train_dataset = windowed_dataset(train_data, ws, bs, sb)
valid_dataset = windowed_dataset(valid_data, ws, bs, sb)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[ws]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(lr=5e-5),
    metrics=['accuracy', 'mae']
)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8*10**(epoch/20)
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)

history = model.fit(
    train_dataset,
    epochs=150,
    callbacks=[tensorboard],
    validation_data=valid_dataset
)

# plt.figure(figsize=(20, 10))
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 30])
# plt.show()
#
#
# plt.figure(figsize=(20, 10))
# plt.semilogx(history.history["lr"], history.history["mae"])
# plt.axis([1e-8, 1e-4, 0, 30])
# plt.show()


mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(len(mae))

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(20, 10))
plt.plot(epochs, mae, label=['Training MAE'])
plt.plot(epochs, val_mae, label=['Validation MAE'])
plt.show()


plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, label=['Training Loss'])
plt.plot(epochs, val_loss, label=['Validation Loss'])
plt.show()


forecast = []
for time in range(len(series)-ws):
    pred_data = series[time:time+ws][np.newaxis]
    if len(pred_data[0]) < ws:
        continue
    prediction = model.predict(pred_data)
    forecast.append(prediction)


forecast = forecast[len(train_df)-ws:]
results = np.array(forecast)[:, 0, 0]

print(len(results))
print(len(valid_data))

plt.figure(figsize=(20, 10))
plt.plot(valid_time, valid_data, label=['Original Data'])
plt.plot(valid_time, results, label=['Prediction Data'])
plt.show()


d = "3557.033333,2848.9833329999997,1294.0805560000001,19.42222222,0.0,0.0,0.0,0.0,0.0,3816.338889,0.0,0.0,3789.961111,1224.0777779999999,17.77777778,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3189.275".split(',')
pr_data = [float(s) for s in d]
print(pr_data)
print(model.predict([pr_data]))


print(tf.keras.metrics.mean_absolute_error(valid_data, results).numpy())
