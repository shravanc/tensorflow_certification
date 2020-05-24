import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = '/tmp/cats_and_dogs_filtered/train'
valid_path = '/tmp/cats_and_dogs_filtered/validation'

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    1/255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    1/255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_generator = valid_gen.flow_from_directory(
    valid_path,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    metrics=['accuracy']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
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
plt.title('Training and Validation Accuracy')

plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, 'bo', label=['Training Loss'])
plt.plot(epochs, val_loss, 'b', label=['Validation Loss'])
plt.title('Training and Validation Loss')
plt.show()
