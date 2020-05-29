import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = '/tmp/rps/'
valid_path = '/tmp/rps-test-set/'


# count = len(os.listdir(valid_path + '/scissors'))
# print(count)

train_count = 2520
valid_count = 372

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
    rotation_range=40,
    fill_mode='nearest'
)
train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=126,
    class_mode='categorical'
)

valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.
)
valid_generator = valid_gen.flow_from_directory(
    valid_path,
    target_size=(150, 150),
    batch_size=126,
    class_mode='categorical'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
    metrics=['accuracy']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=3
)
