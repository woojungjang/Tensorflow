from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

sample = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

idx2char = list(set(sample))
char2idx = {w: i for i, w in enumerate(idx2char)}

dictionary_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
sequence_length = 10  # 임의의 숫자로 지정했다.
learning_rate = 0.1

print('idx2char :', idx2char)
print('char2idx :', char2idx)
print('dictionary_size :', dictionary_size)
print('hidden_size :', hidden_size)
print('num_classes :', num_classes)
print('sequence_length :', sequence_length)

dataX = []
dataY = []
for i in range(0, len(sample) - sequence_length):
    x_str = sample[i:i + sequence_length]
    y_str = sample[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char2idx[c] for c in x_str]  # x str to index
    y = [char2idx[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)
print('batch_size :', batch_size)
print('dataX :', dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
x_one_hot = tf.one_hot(X, num_classes)
print(x_one_hot)  # check out the shape

cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(
   cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([idx2char[t] for t in index]), l)

# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sample
        print(''.join([idx2char[t] for t in index]), end='')
    else:
        print(idx2char[index[-1]], end='')