import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)


train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []
for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

testing_sentences = []
testing_labels = []

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

final_training_labels = np.array(training_labels)
final_testing_labels = np.array(testing_labels)

vocab_size = 10000
oov_token = '<OOV>'
embed_dim = 16
max_len = 120
trunc_type = 'post'

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_token
)
tokenizer.fit_on_texts(training_sentences)

tr_sequences = tokenizer.texts_to_sequences(training_sentences)
tr_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
    tr_sequences,
    maxlen=max_len,
    padding='post',
    truncating=trunc_type
)

val_sequences = tokenizer.texts_to_sequences(testing_sentences)
val_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
    val_sequences,
    padding='post',
    maxlen=max_len,
    truncating=trunc_type
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
    metrics=['accuracy']
)

history = model.fit(
    tr_padded_seq,
    final_training_labels,
    epochs=10,
    validation_data=(val_padded_seq, final_testing_labels),
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.figure(figsize=(20,10))
plt.plot(epochs, acc, label=['Training Accuracy'])
plt.plot(epochs, val_acc, label=['Validation Accuracy'])
plt.show()

