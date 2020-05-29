import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv'
df = pd.read_csv(url)
mean = df['Births'].mean()
std = df['Births'].std()
df['series'] = (df['Births']/mean)/std
print(df.head(2))


series = df['series'].values
time = range(len(series))

plt.figure(figsize=(20,10))
plt.plot(time, series, label=['Female Births'])
plt.title('Time Series Data')
#plt.show()

print(len(series))
total_data = 365
split_time = 300

train_data = series[:split_time]
valid_data = series[split_time:]

time_train = time[:split_time]
time_valid = time[split_time:]

plt.figure(figsize=(20,10))
plt.plot(time_train, train_data, label=['Train Data'])
plt.plot(time_valid, valid_data, label=['Valid Data'])
#plt.show()

ws = 4
bs = 6
sb = 100


def windowed_dataset(t_series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(t_series)
    dataset = dataset.window(ws+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


train_set = windowed_dataset(train_data, ws, bs, sb)
valid_set = windowed_dataset(valid_data, ws, bs, sb)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=[ws], activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(lr=0.001),
    metrics=['accuracy', 'mse']
)

history = model.fit(
    train_set,
    epochs=100,
    validation_data=valid_set
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(20,10))
plt.plot(epochs, acc, label=['Training Accuracy'])
plt.plot(epochs, val_acc, label=['Validation Accuracy'])
plt.title('Training Accuracy')


plt.figure(figsize=(20,10))
plt.plot(epochs, loss, label=['Training Loss'])
plt.plot(epochs, val_loss, label=['Validation Loss'])
plt.title('Training Loss')
#plt.show()


print("***FORECASTING***")
forecast = []
for time in range(len(series-ws)):
    pred_data = series[time:time+ws][np.newaxis]
    if len(pred_data[0]) < ws:
        continue
    predictions = model.predict(pred_data)
    forecast.append(predictions)

print("***FORECASTING***")
forecast = forecast[split_time-ws:]
print(forecast)
results = np.array(forecast)[:, 0, 0]

print(len(valid_data))
print(len(results))


plt.figure(figsize=(20,10))
plt.plot(time_valid, valid_data, label=['Validation Data'])
plt.plot(time_valid, results[:-1], label=['Prediction Data'])
#plt.show()