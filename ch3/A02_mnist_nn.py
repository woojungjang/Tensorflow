# A02_mnist_nn.py
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

w1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

w3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
H = tf.matmul(l2, w3) + b3

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
        feed_dict = {x: batch_xs, y: batch_ys}
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
      x: mnist.test.images, y: mnist.test.labels}))

# Get one and predict
randItem = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[randItem:randItem + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(H, 1), feed_dict={x: mnist.test.images[randItem:randItem + 1]}))

plt.imshow(mnist.test.images[randItem:randItem + 1].
          reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()