import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
df = df.drop('index', axis=1)
print(df.head(2))

df = pd.get_dummies(df, columns=['research'])
print(df.head(2))

train_data = df.sample(frac=0.8, random_state=0)
valid_data = df.drop(train_data.index)

TARGET = 'admit'
train_labels = train_data.pop(TARGET)
valid_labels = valid_data.pop(TARGET)


def norm(raw_df):
    features = raw_df.keys()

    for feature in features:
        if feature == 'research_0' or feature == 'research_1':
            continue
        raw_df[feature] = stats.boxcox(raw_df[feature])[0]


norm(train_data)
norm(valid_data)

train_set = train_data.to_numpy()
valid_set = valid_data.to_numpy()


def windowed_dataset(d_set, labels, batch_size, shuffle_buffer=100):
    dataset = tf.data.Dataset.from_tensor_slices((d_set, labels))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(1)
    return dataset


bs = 32
train_dataset = windowed_dataset(train_set, train_labels, bs)
valid_dataset = windowed_dataset(valid_set, valid_labels, bs)

input_shape = len(train_data.keys())
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[input_shape]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    metrics=['accuracy', 'mae']
)


history = model.fit(
    train_dataset,
    epochs=200,
    validation_data=valid_dataset
)


mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(len(mae))

plt.figure(figsize=(20,10))
plt.plot(epochs, mae, label=['Training MAE'])
plt.plot(epochs, val_mae, label=['Validation MAE'])
plt.show()


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(20,10))
plt.plot(epochs, loss, label=['Training Loss'])
plt.plot(epochs, val_loss, label=['Validation Loss'])
plt.show()