import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


batch = 1
input_height = 3
input_width = 3
input_channels = 2

data = range( input_height* input_width * input_channels)
image = np.reshape(data, ([batch, input_height, input_width, input_channels]))
image = image.astype(np.float32)

plt.imshow(image.reshape(3,3), cmap='Greys', interpolation='nearest')
plt.show()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
filter_height = 2
filter_width = 2
out_channels = 1

w= tf.constant(([1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0]))
w=tf.shape(w,[filter_height,filter_width,input_channels,out_channels])
stride_width=1
stride_height=1

conv2d = tf.nn.conv2d(image,w,stride=[1,stride_width, stride_height,1], padding='VALID')
print(sess.run(w))
conv2d_img = sess.run((conv2d))
print('\nimage:',image)
print()