'''
Created on 2018. 1. 2.

@author: acorn
'''
import tensorflow as tf
import numpy as np
mylist = [[1,2,3],[4,5,6]]
row = tf.argmax(mylist, axis=1)
print(row)


mylist = [[11,6,3],[18,9,2]]

row = tf.argmax(mylist, axis = 1)
print(row)
column = tf.argmax(mylist, axis = 0)
sess = tf.Session()

print(column)

test=[np.exp(1.0),np.exp(0.1)]
result = tf.argmax(test,axis=0)
print(sess.run(result))

row0 = np.array(row)

print(row[: , :])



#a = np.array(row[0])
#a = row[:,0]

print('row0:', row0)
#b = np.array(row[1])
#print(b)
#c = np.exp(a)+np.exp(b)
#print(c)