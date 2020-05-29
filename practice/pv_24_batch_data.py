import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import time

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for physical_device in physical_devices:
#     tf.config.experimental.set_memory_growth(physical_device, True)

plot_path = '/home/shravan/tensorflow_certification/practice/plots'

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/one_year_timeseries.csv'
#url = 'https://raw.githubusercontent.com/shravanc/datasets/master/nov_time_series.csv'
df = pd.read_csv(url)

df[df < 0] = df['series'].mean()
df['n_series'] = df['series'] + 10

raw_df = df.copy()

train_df = df.sample(frac=.99)
valid_df = df.drop(train_df.index)

train_time = range(len(train_df))
valid_time = range(len(valid_df))

print(len(train_time))
print(len(valid_time))


def norm(data):
    data['n_series'] = stats.boxcox(data['n_series'])[0]


norm(train_df)
norm(valid_df)
norm(raw_df)

series = raw_df['n_series'].values
train_data = train_df['n_series'].values
valid_data = valid_df['n_series'].values

bs = 12
ws = 24
sb = 100
shift = 1


def windowed_dataset(t_series, window_size, batch_size, shuffle_buffer, shift_size):
    dataset = tf.data.Dataset.from_tensor_slices(t_series)
    dataset = dataset.window(window_size + 1, shift=shift_size, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


train_dataset = windowed_dataset(
    train_data,
    ws,
    bs,
    sb,
    shift
)

valid_dataset = windowed_dataset(
    valid_data,
    ws,
    bs,
    sb,
    shift
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[ws]),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(shift)
])


lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-8*10**(epochs / 20)
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
    metrics=['mae']
)

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=valid_dataset,
    callbacks=[tensorboard]
)

# image = os.path.join(plot_path, str(round(time.time())))
# plt.figure(figsize=(20, 10))
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 30])
#
# plt.savefig(image)

mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(len(mae))

loss = history.history['loss']
val_loss = history.history['val_loss']

image = os.path.join(plot_path, str(round(time.time())))
plt.figure(figsize=(20, 10))
plt.plot(epochs, mae, label=['Training MAE'])
plt.plot(epochs, val_mae, label=['Validation MAE'])
plt.savefig(image)


image = os.path.join(plot_path, str(round(time.time())))
plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, label=['Training Loss'])
plt.plot(epochs, val_loss, label=['Validation Loss'])
plt.savefig(image)


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

import time
image = os.path.join(plot_path, str(round(time.time())))
plt.figure(figsize=(20, 10))
plt.plot(valid_time, valid_data, label=['Original Data'])
plt.plot(valid_time, results, label=['Prediction Data'])
plt.savefig(image)


d = "3557.033333,2848.9833329999997,1294.0805560000001,19.42222222,0.0,0.0,0.0,0.0,0.0,3816.338889,0.0,0.0,3789.961111,1224.0777779999999,17.77777778,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3189.275".split(',')
pr_data = [float(s) for s in d]
print(pr_data)
print(model.predict([pr_data]))


path = '/tmp/20_epochs/'
model.save(path)
# tf.saved_model.save(model, path)