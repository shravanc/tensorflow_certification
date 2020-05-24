import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.
test_images = test_images / 255.

train_len = len(train_labels)
test_len = len(test_labels)
image_width = 28
image_height = 28
color_dim = 1

train_images = train_images.reshape(train_len, image_width, image_height, color_dim)
test_images = test_images.reshape(test_len, image_width, image_height, color_dim)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    train_labels,
    epochs=10
)

evaluate = model.evaluate(
    test_images,
    test_labels
)

print(evaluate)
