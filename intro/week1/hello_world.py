import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])
model.compile(loss='mse',optimizer=tf.keras.optimizers.SGD())

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)


print(model.predict([10.0]))