#http://cafe.naver.com/ugcadman/1690
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


train_data = np.loadtxt('C:\\Users\\acorn\\Desktop\\train02.csv', dtype=np.float32, delimiter=',')
test_data = np.loadtxt('C:\\Users\\acorn\\Desktop\\test02.csv', dtype=np.float32, delimiter=',')

x_data = train_data[:, 0:2]
print(x_data)
y_data = train_data[:, 2:]
print(y_data)
x_test = test_data[:, 0:2]
print(x_test)
y_test = test_data[:, 2:]
print(y_test)


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

x_data = normalize(x_data)
y_data = normalize(y_data)

x = tf.placeholder( tf.float32, shape=[None, 2])
y = tf.placeholder( tf.float32, shape=[None, 1])
 
w = tf.Variable( tf.random_normal([2, 1], dtype=tf.float32))
b = tf.Variable( 0.0 )
 
H = tf.sigmoid( tf.matmul( x, w ) + b ) 
 
diff = y * tf.log( H ) + (1-y)* tf.log(1-H)
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
 
for step in range(1000):
    _t, _w, _c, _h, _p, _a  = sess.run([train, w, cost, H, predicted, accuracy],
                      feed_dict = {x : x_data, y : y_data } )
    if step % 100 == 0 :
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

mylist = []
W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1 
    curr_cost, curr_W = sess.run([cost, w], feed_dict={w: feed_W})
    if i % 5 == 0:
        sublist = []
        sublist.append(curr_W)
        sublist.append(curr_cost)
        mylist.append( sublist )
    W_val.append(curr_W)
    cost_val.append(curr_cost)
for item in mylist:
    print(item)
    
plt.plot( W_val, cost_val )
plt.ylim(-0.1, 1.1)
plt.show()


