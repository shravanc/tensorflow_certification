import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

dataset = tf.data.Dataset.range(10)
for val in dataset:
  print(val.numpy())


dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
  print(window.numpy())


# separating features and label:
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x, y in dataset:
  print(x.numpy(), y.numpy())


# separating features and label with shuffle


window_size = 5
label_count = 1 # number of labels in the dataset
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(window_size, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size))
dataset = dataset.map(lambda window: (window[: -label_count], window[-label_count:]))
dataset = dataset.shuffle(buffer_size=10)
for x, y in dataset:
  print(x.numpy(), y.numpy())



dataset = tf.data.Dataset.range(10)
dataset = dataset.window(window_size, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
