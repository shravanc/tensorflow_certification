import pandas as pd
import tensorflow as tf
from lib.utils import plot_series
import numpy as np
import matplotlib.pyplot as plt

path = "/home/shravan/python_programs/time_series/test_time_series.csv"
new_file = "/home/shravan/python_programs/time_series/f1.csv"
new_file = "/home/shravan/python_programs/time_series/f2.csv"


df = pd.read_csv(path, names=["series"])
series = df['series'].values
time   = list(range(0, len(series)))

values = df['series'].values
dataset = tf.data.Dataset.range(1000)
#dataset = tf.data.Dataset.from_tensor_slices(values)
dataset = dataset.window(25, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(25))
dataset = dataset.map(lambda window: (window[:-1], window[-1: ]))



fp = open(new_file, 'w+')
for x, y in dataset:
  a = x.numpy().tolist()
  a = ','.join([str(i) for i in a])
  a += f",{str(y.numpy()[0])}"
  print(a)
  print(x.numpy(), y.numpy())
  fp.write(a + '\r\n')
  #break

fp.close()
