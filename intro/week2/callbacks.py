import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.
test_images = test_images / 255.


class CallBacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.6:
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


callback = CallBacks()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(
    train_images,
    train_labels,
    epochs=500,
    callbacks=[callback]
)

evaluate = model.evaluate(
    test_images,
    test_labels
)
print(evaluate)
