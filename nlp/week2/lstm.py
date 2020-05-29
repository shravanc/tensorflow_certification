import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_data, valid_data = imdb['train'], imdb['test']

train_set = []
train_labels = []
for s, l in train_data:
    train_set.append(str(s.numpy()))
    train_labels.append(l.numpy())

valid_set = []
valid_labels = []
for s, l in valid_data:
    valid_set.append(str(s.numpy()))
    valid_labels.append(l.numpy())

vocab_size = 10000
oov_token = "<OOV>"
trunc_type = "post"
max_len = 120
padding = "post"
embed_dim = 16

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_token
)
tokenizer.fit_on_texts(train_set)
word_index = tokenizer.word_index

tr_sequences = tokenizer.texts_to_sequences(train_set)
tr_padded = tf.keras.preprocessing.sequence.pad_sequences(
    tr_sequences,
    padding=padding,
    truncating=trunc_type,
    maxlen=max_len
)

val_sequences = tokenizer.texts_to_sequences(valid_set)
val_padded = tf.keras.preprocessing.sequence.pad_sequences(
    val_sequences,
    padding=padding,
    truncating=trunc_type,
    maxlen=max_len
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu')),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

tr_labels = np.array(train_labels)
val_labels = np.array(valid_labels)

history = model.fit(
    tr_padded,
    tr_labels,
    epochs=10,
    validation_data=(val_padded, val_labels),
)
