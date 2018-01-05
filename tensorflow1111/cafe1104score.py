# csvReaderEx01.py
# csv 파일에서 데이터 읽어 오기
import tensorflow as tf
import numpy as np
 
# https://github.com/hunkim/DeepLearningZeroToAll 에서 csv 다운 받기
 
# loadtxt : 데이터를 튜플 형식으로 반환해준다.
data = np.loadtxt('./score2.csv', dtype=np.float32, delimiter=',')
# print(data)
print(data.shape) #(25, 4)
 
# data.shape는 tuple 자료형인데, 인덱싱이 가능하다.
# table_row = data.shape[0]
# print(table_row)
table_col = data.shape[1]
# print(table_col)
 
column = table_col - 1 # 입력 데이터의 컬럼 갯수
 
# 모든 행의 앞 3개는 입력으로 본다.
x_data = data[:, 0:column]
 
# 모든 행의 맨 뒤 1개는 출력으로 본다.
y_data = data[:, column:(column+1)]
 
print(x_data.shape) #(25, 3) 입력용
print(y_data.shape) #(25, 1) 출력용 으로 나눠봄
x_test = [[20,40,50],[90,88,80]] # 2행 3열
 
x = tf.placeholder( tf.float32, shape=[None, column]) # 25행 3열
y = tf.placeholder( tf.float32, shape=[None, 1]) # 25행 1열
w = tf.Variable( tf.random_normal([column, 1])) # 3행 1열
b = tf.Variable( 0.0 )
 
# H = (25행 3열)*(# 3행 1열)==>(# 25행 1열)
H = tf.matmul(x, w) + b
 
diff = tf.square( H - y )
cost = tf.reduce_mean( diff )#모든 샘플의 오류 평균
 
learn_rate = 1e-5
optimizer= tf.train.GradientDescentOptimizer( learning_rate = learn_rate )
train = optimizer.minimize( cost ) #학습
 
sess = tf.Session()
sess.run( tf.global_variables_initializer() )# 변수 초기화
 
for step in range(5000) :
    _t, _w, _c, _h = sess.run([train, w, cost, H], feed_dict={ x : x_data, y : y_data })
    if step % 100 == 0:
        print(" step :%d, loss:%f " % (step, _c))
        # print('H : ', _h)
 
# x_test는 2행 3열이므로, (2행 3열)*(# 3행 1열)==>(# 2행 1열)이 출력이 된다.
result = sess.run( H, feed_dict={ x : x_test })
print('score predict', result) # 예측 결과