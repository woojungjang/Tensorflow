# 붓꽃 데이터 예측하기
import tensorflow as tf
import numpy as np
 
data = np.loadtxt('./iris_two.csv', dtype=np.float32, delimiter=',')
# print(data)
print(data.shape)
table_col = data.shape[1]
print(table_col)
column = table_col - 1 # 입력 데이터의 컬럼 갯수
 
testM = 4 # 테스트를 위한 행 갯수
m = len(data) - testM
x_data = data[0:m, 0:column]
y_data = data[0:m, column:(column+1)]
x_test = data[m:, 0:column]
y_test = data[m:, column:(column+1)]
# print('x_data:', x_data)
# print('y_data:', y_data)
# print('x_test:', x_test)
# print('y_test:', y_test)
 
x = tf.placeholder( tf.float32, shape=[None, column])
y = tf.placeholder( tf.float32, shape=[None, 1])
 
w = tf.Variable( tf.random_normal([column, 1]))
b = tf.Variable( 0.0 )
 
H = tf.sigmoid( tf.matmul( x, w ) + b )
 
diff = y * tf.log(H) + (1 - y) * tf.log(1 - H)
cost = -tf.reduce_mean( diff )
 
learn_rate = 0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate = learn_rate )
train = optimizer.minimize( cost )
 
predicted = tf.cast( H  > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
 
sess=tf.Session()
sess.run(tf.global_variables_initializer())
 
def inlineprint( mylist ):
    imsi = ''
    for item in mylist :
        imsi += str(item) + ' '
    print (imsi)
 
for step in range(10000):
    _t, _w, _c, _h, _p, _a  = sess.run([train, w, cost, H, predicted, accuracy],
                      feed_dict = {x : x_data, y : y_data } )
    if step % 1000 == 0 :
        print('step : %d, cost : %f, accuracy : %f' % (step, _c, _a))
        print('hypothesis', end=' : ')
        inlineprint(_h)
        print('predicted', end=' : ')
        inlineprint(_p)
        print('---------------------------------------------')
 
# predict = sess.run(predicted, feed_dict = { x : x_test, y : y_test })
predict = sess.run(predicted, feed_dict = { x : x_test })
# print('class predict', predict)
 
def getCategory( datalist ):
    mylist = ['Iris-setosa', 'Iris-versicolor']
 
    for item in range(len(datalist)):
        print( datalist[item], mylist[(int)(datalist[item])] )
 
getCategory(predict)

