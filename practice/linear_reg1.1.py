import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)
df = pd.get_dummies(df)

TARGET_COL = 'charges'

train_data = df.sample(frac=0.8, random_state=0)
valid_data = df.drop(train_data.index)


train_stats = train_data.describe()
train_stats.pop(TARGET_COL)
train_stats = train_stats.transpose()

valid_stats = valid_data.describe()
valid_stats.pop(TARGET_COL)
valid_stats = valid_stats.transpose()

train_labels = train_data.pop(TARGET_COL)
train_labels = train_labels.values

valid_labels = valid_data.pop(TARGET_COL)
valid_labels = valid_labels.values


def norm(stats, raw_df):
    return (raw_df/stats['mean'])/stats['std']


norm_train_set = norm(train_stats, train_data)
norm_valid_set = norm(valid_stats, valid_data)

train_set = norm_train_set.to_numpy()
valid_set = norm_valid_set.to_numpy()


print(len(train_set))
print(len(valid_set))


def windowed_dataset(features, labels, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


bs = 32
sb = 200
train_dataset = windowed_dataset(train_set, train_labels, bs, sb)
valid_dataset = windowed_dataset(valid_set, valid_labels, bs, sb)


input_shape = len(df.keys())-1
print(input_shape)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[input_shape]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    metrics=['accuracy', 'mse']
)

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=valid_dataset
)


acc = history.history['accuracy']
print(acc)
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.figure(figsize=(20,10))
plt.plot(epochs, acc, label=['Training Accuracy'])
plt.plot(epochs, val_acc, label=['Validation Accuracy'])
plt.title('Accuracy Plot')
plt.show()