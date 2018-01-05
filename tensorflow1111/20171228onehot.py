import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

y = [3, 0, 4]
nb_classes = 5
onehot = tf.one_hot(y, nb_classes)
sess = tf.Session()
_one = sess.run(onehot)
print('----------------------------')
print(_one)
result = tf.argmax(_one, a)






