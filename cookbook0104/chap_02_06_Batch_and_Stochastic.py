##chap_02_06_Batch_and_Stochastic.py
# Batch and Stochastic Training(쿡북91)
# Stochastic(확률적): 1번에 1개씩 훈련시키는것
# Batch(일괄): 1번에 여러개(batch_size) 넣어서 훈련시키고, 평균비용으로 훈련시키는것(간김에 한꺼번에 일처리)
#(쿡북96에 두가지비교표)
#----------------------------------
#
#  This python function illustrates two different training methods:
#  batch and stochastic training.  For each model, we will use
#  a regression model that predicts one model variable.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# We will implement a regression example in stochastic and batch training

# Stochastic Training: 확률적학습
#100개의 샘플(np.random.normal(1, 0.1, 100))에서 한개씩 뽑아서 학습시킨다. 
#이것을 100번수행(아래 for문에 나오는 100에 해당)한다.
# Create graph
sess = tf.Session()

# Create data
x_vals = np.random.normal(1, 0.1, 100) #평균, 표준편차, 요소갯수
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))

# Add operation to graph
my_output = tf.multiply(x_data, A)

# Add L2 loss operation to graph
loss = tf.square(my_output - y_target)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_stochastic = []
# Run Loop
for i in range(100):
    rand_index = np.random.choice(100) #0-99까지 사이의 임의의 1개 고르기, 샘플링
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)
        
#_________________________________________________________________(확률적학습end)_____________________
# Batch Training: 일괄학습

# Re-initialize graph
ops.reset_default_graph()
sess = tf.Session()

# Declare batch size
batch_size = 20 #한번에 20개씩 넣어서 실행시킬 숫자

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Add operation to graph
my_output = tf.matmul(x_data, A)

# Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(my_output - y_target))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_batch = [] #일괄학습비용함수 저장소
# Run Loop
for i in range(100): #한번에 20개(batch_size)씩 복원추출하기
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)
        
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()