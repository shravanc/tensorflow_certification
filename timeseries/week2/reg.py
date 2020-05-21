import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


url = "https://raw.githubusercontent.com/shravanc/datasets/master/f1.csv"
url = "https://raw.githubusercontent.com/shravanc/datasets/master/f2.1.csv"
raw_dataset = pd.read_csv(url)
dataset = raw_dataset.copy()

print(dataset.tail())


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset  = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop('y')
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('y')
test_labels  = test_dataset.pop('y')


def norm(x):
  print("train_stats---->", train_stats['mean'][0])
  print("train_stats---->", train_stats['std'][0])
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data  = norm(test_dataset)



def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

print(model.summary())


EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('PV Output')
plt.show()

plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('PV Output')

plt.show()


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])


plotter.plot({'Early Stopping': early_history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()


def normal(x):
  print("train_stats---->", train_stats['mean'][0])
  print("train_stats---->", train_stats['std'][0])
  return (x - 490.) / 280.
d = "3557.033333,2848.9833329999997,1294.0805560000001,19.42222222,0.0,0.0,0.0,0.0,0.0,3816.338889,0.0,0.0,3789.961111,1224.0777779999999,17.77777778,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3189.275".split(',')
d = [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
d = [normal(float(s)) for s in d]
print(d)
print("preidiction")
print(model.predict([d]).flatten())
print(model.predict([d]))



