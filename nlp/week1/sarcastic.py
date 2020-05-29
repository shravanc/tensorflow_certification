import tensorflow as tf
import json

path = '/tmp/sarcasm.json'
with open(path) as f:
    datastore = json.load(f)

sentences = []
labels = []

for data in datastore:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])


tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    padding='post',
    maxlen=40,
    truncating='post'
)
print(padded[0])

