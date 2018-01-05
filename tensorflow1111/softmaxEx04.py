# softMaxEx04.py
import tensorflow as tf
import numpy as np
 
data = np.loadtxt('./iris_three.csv', delimiter=',', dtype=np.float32)
 
# print (data.shape)
table_row = data.shape[0]  # 엑셀 파일에서 읽어 들인 데이터의 총 행수
table_col = data.shape[1]  # 엑셀 파일에서 읽어 들인 데이터의 총 열수
column = table_col - 1  # 입력 데이터의 컬럼 갯수
 
testing_row = 3  # 테스트 용 데이터 셋의 행수
training_row = table_row - testing_row  # 훈련용 데이터 셋의 행수
 
nb_classes = 3  # 클래스의 갯수(붓꽃의 종류)
 
print('입력 데이터의 컬럼 갯수(column) : %d' % (column))
print('클래스의 갯수(nb_classes) : %d' % (nb_classes))
print('table_row : %d, training_row : %d' % (table_row, training_row))
 
x_train = data[0:training_row, 0:column]
y_train = data[0:training_row, column:(column + 1)]
 
x_test = data[training_row:, 0:column]
y_test = data[training_row:, column:(column + 1)]
 
print('-------------------------------------------------------------------------')
print('훈련용 입력 데이터(x_train):\n', x_train)
print('-------------------------------------------------------------------------')
print('훈련용 정답 데이터(y_train):\n', y_train)
print('-------------------------------------------------------------------------')
print('테스트용 입력 데이터(x_test):\n', x_test)
print('-------------------------------------------------------------------------')
print('테스트용 정답 데이터(y_test):\n', y_test)
 
x = tf.placeholder(tf.float32, [None, column]) # M행 4열
y = tf.placeholder(tf.int32, [None, 1]) # M행 1열
 
w = tf.Variable(tf.random_normal([column, nb_classes]) ) # 4행 3열
b = tf.Variable(tf.random_normal([nb_classes]) )
 
# 여기서 점검해보기
# y(M행 1열) = x(M행 4열 ) * w(4행 3열)
# x와 w의 행렬 곱이 y와 동일하려면 y는 (M행 3열)이 되어야 한다.
# 따라서, y를 nb_classes(값 : 3)에 대하여 one hot 함수를 적용해줘야 한다.
 
Y_one_hot = tf.one_hot(y, nb_classes)  # one hot
# one_hot 함수를 사용하면 차원(rank)의 크기가 하나 증가되어 생성된다.
 
# 따라서, 아래와 같이 reshape를 이용하여 차원을 변경해 주어야 한다.
# reshape에 사용되는 [-1, nb_classes]는 알아서 shape를 (?, nb_classes)과 같은 형태로 변경시켜 준다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
 
# 다시 점검해보기
# y(M행 3열) = x(M행 4열 ) * w(4행 3열)
 
# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# M행 16열 * 16행 7열 => M행 7열
logits = tf.matmul(x, w) + b
H = tf.nn.softmax( logits )
 
# softmax_cross_entropy_with_logits 함수는 logits와 Y_one_hot을 이용하여 cost를 정의해 주는 함수이다.
# 다음과 같이 사용하면 된다.
diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean( diff )
 
learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)
 
# 컴퓨터가 예측한 값(prediction)
prediction = tf.argmax(H, axis = 1)
 
print('-------------------------------------------------------------------------')
 
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
 
   for step in range(2000): #2000
       sess.run(train, feed_dict={x: x_train, y: y_train})
 
       if step % 100 == 0:
           _yonehot, _c, _h, _p  = \
               sess.run([Y_one_hot, cost, H, prediction], \
                    feed_dict={x: x_train, y: y_train})
           if step == 0 :
                print("one hot 결과\n", _yonehot)  #
                print('-------------------------------------------------------------------------')
 
           print("훈련 회수: {:5}\t비용 : {:.3f}".format(step, _c))
           print("가설\n", _h) #
           print("컴퓨터가 예측한 값(prediction)", _p)  #
           print('-------------------------------------------------------------------------')
 
   pred = sess.run(prediction, feed_dict={x: x_train})
   for p, y in zip(pred, y_train.flatten()):
       print("[{}] Prediction: {}, Answer y: {}".format(p == int(y), p, int(y)))
 
   _h2 = sess.run([H], feed_dict={x: x_test})
   print( np.argmax(_h2, axis=1))
