# A05_mnist_nn_dropout.py
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
dropout_prob = tf.placeholder(tf.float32)

w1 = tf.get_variable("w1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)
l1 = tf.nn.dropout(l1, keep_prob=dropout_prob) #1st layer에 dropout

w2 = tf.get_variable("w2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
l2 = tf.nn.dropout(l2, keep_prob=dropout_prob)

w3 = tf.get_variable("w3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=dropout_prob)

w4 = tf.get_variable("w4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
l4 = tf.nn.relu(tf.matmul(L3, w4) + b4)
l4 = tf.nn.dropout(l4, keep_prob=dropout_prob)

w5 = tf.get_variable("w5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
H = tf.matmul(l4, w5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys, dropout_prob : 0.7}
        _cost, _ = sess.run([cost, train], feed_dict=feed_dict)        
        avg_cost += _cost

    avg_cost = avg_cost / total_batch
    
    print('{:.9f}'.format(avg_cost))
#     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('학습 완료!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={
      x: mnist.test.images, y: mnist.test.labels, dropout_prob : 1}))

# Get one and predict
randItem = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[randItem:randItem + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(H, 1), feed_dict={x: mnist.test.images[randItem:randItem + 1], dropout_prob : 1}))

plt.imshow(mnist.test.images[randItem:randItem + 1].
          reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()