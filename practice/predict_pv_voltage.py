import tensorflow as tf
import numpy as np

path = '/tmp/time_series_model/'
path = '/tmp/20_epochs/'
model = tf.keras.models.load_model(path)


d = "3557.033333,2848.9833329999997,1294.0805560000001,19.42222222,0.0,0.0,0.0,0.0,0.0,3816.338889,0.0,0.0,3789.961111,1224.0777779999999,17.77777778,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3189.275".split(',')
# pr_data = [float(s) for s in d]
# print(pr_data)
# print(model.predict([pr_data]))


window_size = 24
def get_24_values(val):
    pr_data = [float(s) for s in val]
    for time in range(window_size):
        data = list(pr_data[time:time+window_size]) #[np.newaxis]
        prediction = model.predict([data])
        pr_data.append(prediction[0][0])

    return pr_data[window_size:]


result = get_24_values(d)
print(result)