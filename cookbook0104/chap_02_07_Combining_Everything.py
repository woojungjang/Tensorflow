# chap_02_07_Combining_Everything.py
# Combining Everything Together(쿡북96)
# 붓꽃 데이터셋 분류기
# 어떤꽃이 setosa인지 아닌지를 판단
# 꽃잎의 길이와 너비만으로 setosa종인지 아닌지 구분하는 예시
#----------------------------------
# This file will perform binary classification on the
# iris dataset. We will only predict if a flower is
# I.setosa or not.
#
# We will create a simple binary classifier by creating a line
# and running everything through a sigmoid to get a binary predictor.
# The two features we will use are pedal length and pedal width.
#
# We will use batch training, but this can be easily
# adapted to stochastic training.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Load the iris data
# iris.target = {0, 1, 2}, where '0' is setosa
# iris.data ~ [sepal.width, sepal.length, pedal.width, pedal.length]
iris = datasets.load_iris()

#iris.target: 품종정보를 담고있는 컬럼
#setosa 품종은 1로 non-setosa 품종은 0으로 변환
binary_target = np.array([1. if x==0 else 0. for x in iris.target])

#꽃잎의 길이와 너비를 읽어들임
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

# Declare batch size
batch_size = 20 #일괄 학습의 사이즈

# Create graph
sess = tf.Session()

# Declare placeholders
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables A and b (0 = x1 - A*x2 + b)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Add model to graph:
# x1 - A*x2 + b
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Run Loop
for i in range(1000):
    #iris: 쿡북 51쪽 참조
    #iris데이터는 150개의 행을 가지고 있다.
    rand_index = np.random.choice(len(iris_2d), size=batch_size) #len(iris_2d) 150개에서 랜덤하게 20개씩 복원추출
    #rand_x = np.transpose([iris_2d[rand_index]])
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x]) #꽃잎의 길이
    rand_x2 = np.array([[x[1]] for x in rand_x]) #꽃잎의 너비
    #rand_y = np.transpose([binary_target[rand_index]])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
        

# Visualize Results
# Pull out slope/intercept
[[slope]] = sess.run(A) #A=기울기
[[intercept]] = sess.run(b) #y절편

# Create fitted line
x = np.linspace(0, 3, num=50)

#직선을 그리기 위한 점 정보를 담고있는 리스트
ablineValues = [] 

for i in x:
  ablineValues.append(slope*i+intercept)

# Plot the fitted line over the data
# a[0]:길이, a[1]:너비
# binary_target[i]==1가 품종
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1] #enumerate(반복적으로 나열)
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()