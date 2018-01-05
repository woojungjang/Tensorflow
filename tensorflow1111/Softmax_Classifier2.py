#http://cafe.naver.com/ugcadman/1359
import tensorflow as tf
import numpy as np
 
# 중첩 리스트의 열 갯수 구하는 함수
def col_length( input ):
    sublist = input[0]
    length = len(sublist)
    return length
 
def getCategory( datalist ):
    mylist = ['강아지', '고양이', '토끼']
 
    for item in range(len(datalist)):
        print(datalist[item], mylist[(int)(datalist[item])])
 
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
 
y_data = [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
 
column = col_length( x_data)
print( '컬럼수 : ', column )
nb_classes = col_length( y_data )
print( '클래스 갯수 : ', nb_classes )
 
x = tf.placeholder("float", [None, column])
y = tf.placeholder("float", [None, nb_classes])
 
w = tf.Variable(tf.random_normal([column, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
 
H = tf.nn.softmax(tf.matmul(x, w) + b)
 
diff = -tf.reduce_sum(y * tf.log(H), axis=1)
cost = tf.reduce_mean( diff )
 
learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize( cost )
 
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
 
   for step in range(2001):
       sess.run(train, feed_dict={x: x_data, y: y_data})
       if step % 500 == 0:
           _c, _h = sess.run([cost, H], feed_dict={x: x_data, y: y_data})
           print('학습 회수 :', step, ', 비용 :',_c)
           print('가설 :\n', _h)
           print('최대값 인덱스(argmax) :', np.argmax(_h, axis=1))
           print('최종 결과:')
           getCategory(np.argmax(_h, axis=1))
           print('--------------------------------------')
