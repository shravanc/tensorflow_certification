import tensorflow as tf

sentences = [
    'I love my dog',
    'I love my cat',
    'Dog I had was the best!'
]

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=100,
    oov_token='<OOV>'
)
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
pad = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    maxlen=10,
    padding='post',
    truncating='post'
)
print(pad)

test_sentence = [
    'really very good cat'
]

sequences = tokenizer.texts_to_sequences(test_sentence)
pad = tf.keras.preprocessing.sequence.pad_sequences(sequences)
print(pad)