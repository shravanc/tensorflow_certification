import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import zipfile
#
# local_zip = '/tmp/horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/tmp/horse-or-human')
# local_zip = '/tmp/validation-horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/tmp/validation-horse-or-human')
# zip_ref.close()


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for physical_device in physical_devices:
#     tf.config.experimental.set_memory_growth(physical_device, True)

train_path = '/tmp/horse-or-human'
test_path = '/tmp/validation-horse-or-human'

image_width = 300
image_height = 300

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(image_width, image_height),
    batch_size=128,
    class_mode='binary'
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_generator = test_gen.flow_from_directory(
    test_path,
    target_size=(image_width, image_height),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    metrics='accuracy'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=test_generator,
    validation_steps=8
)