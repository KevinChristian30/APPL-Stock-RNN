import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

layer = {
    'input' : 1,
    'output' : 1
}

batch_size = 3
time_step = 3
epoch = 1000

def load_dataset():
    df = pd.read_csv('AAPL.csv', index_col='Date')
    return df

def get_batch(dataset, time_step, batch_size):
    input_batch = np.zeros(shape=(batch_size, time_step, layer['input']))
    output_batch = np.zeros(shape=(batch_size, time_step, layer['output']))
    
    for i in range(batch_size):
        point = np.random.randint(0, len(dataset) - time_step)
        input_batch[i] = dataset[point: point + time_step]
        output_batch[i] = dataset[point + 1: point + time_step + 1]
    
    return input_batch, output_batch

dataset = load_dataset()
split = int(len(dataset) * 0.7)
train_dataset = dataset[:split]
test_dataset = dataset[split:]

scaler = MinMaxScaler().fit(train_dataset)
norm_train = scaler.transform(train_dataset)

cell = tf.nn.rnn_cell.BasicRNNCell(10, activation = tf.nn.relu)

cell = tf.contrib.rnn.OutputProjectionWrapper(cell, layer['output'], activation=tf.nn.relu)

input_feature = tf.placeholder(tf.float32, [None, time_step, layer['input']])
input_target = tf.placeholder(tf.float32, [None, time_step, layer['output']])

output, _ = tf.nn.dynamic_rnn(cell, input_feature, dtype=tf.float32)


loss = tf.reduce_mean(0.5 * (input_target - output) ** 2)

train = tf.train.AdamOptimizer(0.01).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1, epoch + 1):
        input_batch, output_batch = get_batch(norm_train, time_step, batch_size)
        
        feed = {
            input_feature : input_batch,
            input_target : output_batch
        }
        
        sess.run(train, feed_dict = feed)
        
        if i % 100 == 0:
            print("Iteration : {}, Loss : {}".format(i, sess.run(loss, feed_dict=feed)))
    
    saver.save(sess, 'model/rnn_model.ckpt')
    
with tf.Session() as sess:
    seed_data = list(norm_train)
    saver.restore(sess, 'model/rnn_model.ckpt')
    
    for i in range(len(test_dataset)):
        input_batch = np.array(seed_data[-time_step:]).reshape([1, time_step, layer['input']])
        feed = {input_feature : input_batch}
        predict = sess.run(output, feed_dict=feed)
        
        seed_data.append(predict[0, -1, 0])
        
    predict_result = scaler.inverse_transform(np.array(seed_data[-len(test_dataset) :]).reshape([len(test_dataset), 1]))
    test_dataset['Prediction'] = predict_result
    test_dataset.plot()
    plt.show()
