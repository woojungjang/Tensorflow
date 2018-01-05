#http://cafe.naver.com/ugcadman/1690
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_data = np.loadtxt('C:\\Users\\acorn\\Desktop\\train01.csv', dtype=np.float32, delimiter=',')
test_data = np.loadtxt('C:\\Users\\acorn\\Desktop\\test01.csv', dtype=np.float32, delimiter=',')



x_data = train_data[:, 0:2]
y_data = train_data[:, 1:]
x_test = test_data[:, 0:2]
y_test = train_data[:, 1:]

def sigmoid( x_data ):
    return 1 / ( 1 + np.exp(-x_data))
x = np.arange(x_data)

y = sigmoid( x )
print(y)

plt.plot( x, y )
plt.ylim(-0.1, 1.1)
plt.show()
 
def inlineprint( mylist ):
    imsi = ''
    for item in mylist :
        imsi += str(item) + ' '
    print (imsi)
    
H = np.array([[0.5], [0.6], [0.7], [0.8]])
 
print('일반적인 출력 형식')
print( H )
print('\n인라인 출력하기')
inlineprint(H)

inlineprint(x)
print(x)

print(train_data.shape)
#(45, 2)
print(test_data.shape)
#(5, 2)
def normalize(input):
    max = np.max(input, 0)
    min = np.min(input, 0)
    out = (input-min)(max-min)
    print(out)
    return out

#x_data = normalize(x_data)
#y_data = normalize(y_data)

x = tf.placeholder( tf.float32, shape=[None, 2])
y = tf.placeholder( tf.float32, shape=[None, 1])
 
w = tf.Variable( tf.random_normal([2, 1], dtype=tf.float32))
b = tf.Variable( 0.0 )
 
H = tf.matmul( x, w ) + b 
 
diff = tf.square(y-H)
cost = tf.reduce_mean( diff )
 
learn_rate = 1e-5
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
 
for step in range(100):
    _t, _w, _c, _h, _p, _a  = sess.run([train, w, cost, H, predicted, accuracy],
                      feed_dict = {x : x_data, y : y_data } )
    if step % 10 == 0 :
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
    mylist = ['x', 'y']
 
    for item in range(len(datalist)):
        print( datalist[item], mylist[(int)(datalist[item])] )
 
getCategory(predict)


