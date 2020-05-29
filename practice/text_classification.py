import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/datasets_2050_3494_SPAM%20text%20message%2020170820%20-%20Data.csv'
df = pd.read_csv(url)
print(df.head(2))
mapper = {'ham': 0, 'spam': 1}
df['Category'] = df['Category'].replace(mapper)
print(df.head(2))

texts = df['Message'].values
label = df['Category'].values

split_time = 5000

train_data = texts[:split_time]
valid_data = texts[split_time:]

train_label = label[:split_time]
valid_label = label[split_time:]


train_label = np.array(train_label)
valid_label = np.array(valid_label)


oov_token = '<OOV>'
trunc_type= 'post'
padding = 'post'
vocab_size = 1000
max_len = 120
embed_dim = 16

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_token
)

tokenizer.fit_on_texts(train_data)
train_seq = tokenizer.texts_to_sequences(train_data)
train_pad = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq,
    truncating=trunc_type,
    padding=padding,
    maxlen=max_len
)

valid_seq = tokenizer.texts_to_sequences(valid_data)
valid_pad = tf.keras.preprocessing.sequence.pad_sequences(
    valid_seq,
    truncating=trunc_type,
    padding=padding,
    maxlen=max_len
)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(
    train_pad,
    train_label,
    epochs=100,
    validation_data=(valid_pad, valid_label)
)

