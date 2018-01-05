'''
Created on 2017. 12. 22.

@author: acorn
'''
import tensorflow as tf

# 상수 정의하기 - 1
a = tf.constant(120, name='a')
b = tf.constant(130, name='b')

# 변수 정의 및 초기 값 할당 - 2
v = tf.Variable(0, name='v')

# 데이터 플로우 그래프 정의하기 - 3
result = a + b 

# 세션 실행하기 - 4
sess = tf.Session()

# v의 내용 출력하기 -- 5 
print('결과:', sess.run(result))

import tensorflow as tf

# 상수 정의하기 - 1
a = tf.constant(120, name='a')
b = tf.constant(130, name='b')

# 변수 정의 및 초기 값 할당 - 2
v = tf.Variable(0, name='v')

# 데이터 플로우 그래프 정의하기 - 3
result = a + b 

# 세션 실행하기 - 4
sess = tf.Session()

# v의 내용 출력하기 -- 5 
print('결과:', sess.run(result))


a = tf.constant(14)
b = tf.constant(5)
add = a + b
sub = a - b
sess = tf.Session()
print('더하기:', sess.run(add))
print('빼기:', sess.run(sub))


######################################
#PlaceHolder
a = tf.placeholder(tf.int32, [None])
b = tf.constant(10)
sess = tf.Session()
r1 = sess.run(x10_op, feed_dict={a:[1,2,3,4,5]})
print(r1)

r2=sess.run(x10_op, feed_dict)
###################################

x1 = [1,2,3]
x2 = [2,3,4]
a = tf.placeholder(tf.int32, [None])
b = tf.constant(2)
 
x2_op = a * b
sess = tf.Session()

result = [?,?,?]
#############################################
