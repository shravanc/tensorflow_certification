import tensorflow as tf
import tensorflow_datasets as tfds
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = '/home/shravan/Downloads/358221_702372_bundle_archive/indoorCVPR_09/Images/classification_data/train'
valid_path = '/home/shravan/Downloads/358221_702372_bundle_archive/indoorCVPR_09/Images/classification_data/valid'

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(1/255.)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(1/255.)

# print('res--')
# path = os.path.join(train_path, 'restaurant')
# print(len(os.listdir(path)))
#
# path = os.path.join(valid_path, 'restaurant')
# print(len(os.listdir(path)))
# print('res--')
#
# print('bar--')
# path = os.path.join(train_path, 'bar')
# print(len(os.listdir(path)))
#
# path = os.path.join(valid_path, 'bar')
# print(len(os.listdir(path)))
# print('bar--')
#
# print('bedroom--')
# path = os.path.join(train_path, 'bedroom')
# print(len(os.listdir(path)))
#
# path = os.path.join(valid_path, 'bedroom')
# print(len(os.listdir(path)))
# print('bedroom--')


total_train_samples = 1380
total_valid_samples = 150


train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(300,300),
    batch_size=12,
    class_mode='categorical'
)

valid_generator = valid_gen.flow_from_directory(
    valid_path,
    target_size=(300,300),
    batch_size=10
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=115,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=15
)
