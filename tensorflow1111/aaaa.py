import tensorflow as tf
sess = tf.constant(1.0)
print('a = ', a, '/값:', sess.run(a))
b=tf.constant(2.0, dtype=tf.float32)
print('b=', b,'/값', sess.run(b))
