import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

sample = "if you want you i want to meet you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dictionary_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)

print('idx2char :', idx2char) # 유니크한 문자열을 저장하고 있는 리스트
print('char2idx :', char2idx) # {문자:인덱스} 형식을 저장하고 있는 사전 
print('dictionary_size :', dictionary_size) # 
print('hidden_size :', hidden_size) # 
print('num_classes :', num_classes) # idx2char와 동일한 값
print('batch_size :', batch_size) # 사전의 크기
print('sequence_length :', sequence_length) # 

learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index
print('sample_idx :', sample_idx) # 각 글자들의 색인을 저장하고 있는 리스트

x_data = [sample_idx[:-1]]  # x data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # y label sample (1 ~ n) hello: ello

x = tf.placeholder(tf.int32, [None, sequence_length])  # x data
y = tf.placeholder(tf.int32, [None, sequence_length])  # y label

x_one_hot = tf.one_hot(x, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# # FC layer
# X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# 
# # reshape out for sequence_loss
# outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        l, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
        result = sess.run(prediction, feed_dict={x: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))