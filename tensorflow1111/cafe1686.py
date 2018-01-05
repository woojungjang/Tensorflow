import tensorflow as tf
import numpy as np

# loadtxt : 데이터를 튜플 형식으로 반환해준다.
data = np.loadtxt('./sample_data.csv', dtype=np.float32, delimiter=',')
data =data[ : , 1:]
# print(data)
print(data.shape) #(150, 4)
 
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

def normalize(input):
    max = np.max(input,0)
    min = np.min(input,0)
    out = (input-min)/(max-min)
    print(out)
    return out


testM = 10 
m = len(x_data) - testM
print('m', m)

x_data = normalize(x_data)
y_data = normalize(y_data)
x_data = x_data[ 0:m, : ]
y_data = y_data[ 0:m, : ]
x_test = x_data[ m:, : ]
y_test = y_data[ m:, : ]

x = tf.placeholder( tf.float32, shape=[None, column]) # 25행 3열
y = tf.placeholder( tf.float32, shape=[None, 1]) # 25행 1열
w = tf.Variable( tf.ones([column, 1], dtype=tf.float32)) # 3행 1열
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
 
epoch = 10000
for step in range(epoch) :
    _t, _w, _c, _h = sess.run([train, w, cost, H], feed_dict={ x : x_data, y : y_data })
    if step % (epoch/10) == 0:
        print(" step :%d, loss:%f " % (step, _c))
        # print('H : ', _h)
# x_test = [[1,1,3.2], [1,1,3.2]]
# x_test = normalize(x_test)
# x_test는 2행 3열이므로, (2행 3열)*(# 3행 1열)==>(# 2행 1열)이 출력이 된다.
result = sess.run( H, feed_dict={ x : x_test })
print('input', x_test)
print(result)
print('y_test', y_test)
print('why', sess.run(w))
print('result predict', result) # 예측 결과