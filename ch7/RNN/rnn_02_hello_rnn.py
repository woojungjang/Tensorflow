# Lab 12 RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

# Teach hello: hihell -> ihello
# h를 넣으면 i, i를 넣으면 h가 나오는 문제

# 유니크한 문자들을 저장하고 있는 리스트
idx2char = ['o', 'h', 'm', 'y', 'k', 'r', 'e', 'a']

x_data = [[0, 1, 2, 3, 4, 0, 5, 6]]   # ohmykore

# x_one_hot는 x_data를 one hot 인코딩해놓은 데이터이다.
x_one_hot = [[
              [1, 0, 0, 0, 0, 0, 0, 0],   # o 0
              [0, 1, 0, 0, 0, 0, 0, 0],   # h 1
              [0, 0, 1, 0, 0, 0, 0, 0],   # m 0
              [0, 0, 0, 1, 0, 0, 0, 0],   # y 0
              [0, 0, 0, 0, 1, 0, 0, 0],   # k 0
              [1, 0, 0, 0, 0, 0, 0, 0],   # r 0
              [0, 0, 0, 0, 1, 0, 0, 0],   # e 2
              [0, 0, 0, 0, 0, 1, 0, 0]]]  # a 3

# 출력 값(ihello)
y_data = [[1, 2, 3, 4, 0, 5, 6, 7]]  

input_dimension = 8  # one-hot의 크기 입력
hidden_size = 8  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 8  # |ihello| == 6
learning_rate = 0.1

x = tf.placeholder(
    tf.float32, [None, sequence_length, input_dimension])  # x one-hot
y = tf.placeholder(tf.int32, [None, sequence_length])  # y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

# 초기 상태를 저장하고 있는 객체
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(
    cell, x, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
print('weights :' , weights)

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=y, weights=weights)

loss = tf.reduce_mean(sequence_loss)

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={x: x_one_hot, y: y_data})
        result = sess.run(prediction, feed_dict={x: x_one_hot})
        print('\nstep :', i, "loss:", l, "\nprediction: ", result, "\nAnswer: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("Prediction string : ", ''.join(result_str))