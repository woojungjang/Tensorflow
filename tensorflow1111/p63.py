import tensorflow as tf
y = [0, 1, 2]
nb_classes = 4
onehot = tf.one_hot(y, nb_classes)
print(onehot)
sess = tf.Session()
print( sess.run(onehot)) 
_one = sess.run(onehot)
print(_one)
result = tf.argmax( _one, axis=1)
result = sess.run(result)
print(result)


x = [2.0, 1.0. 0.1]
