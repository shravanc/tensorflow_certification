import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = '/tmp/cats_and_dogs_filtered/train'
valid_path = '/tmp/cats_and_dogs_filtered/validation'

trcount = 2000
vlcount = 1000

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=40,
    fill_mode='nearest'
)
train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.
)
valid_generator = valid_gen.flow_from_directory(
    valid_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# ===Model
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=50
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(20, 10))
plt.plot(epochs, acc, 'bo', label=['Training Accuracy'])
plt.plot(epochs, val_acc, 'b', label=['Validation Accuracy'])
plt.title('Accuracy Plot')

plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, 'bo', label=['Training Loss'])
plt.plot(epochs, val_loss, 'b', label=['Validation Loss'])
plt.title('Loss Plot')
plt.show()
