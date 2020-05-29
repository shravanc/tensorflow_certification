import pandas as pd
import tensorflow as tf

url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)

print(df.head(2))
print(df.isna().sum())

df = pd.get_dummies(df)
print(df.head(2))

train_df = df.sample(frac=0.8, random_state=0)
valid_df = df.drop(train_df.index)

print(len(train_df))
print(len(valid_df))

train_stats = train_df.describe()
train_stats.pop("charges")
train_stats = train_stats.transpose()

valid_stats = valid_df.describe()
valid_stats.pop("charges")
valid_stats = valid_stats.transpose()

train_ys = train_df.pop('charges')
valid_ys = valid_df.pop('charges')


def normalise(x, stats):
    return (x - stats['mean']) / stats['std']


train_set = normalise(train_df, train_stats)
valid_set = normalise(valid_df, valid_stats)

print(df.head(3))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy']
)

model.fit(
    train_set,
    train_ys,
    epochs=10,
    validation_data=(valid_set, valid_ys)
)
