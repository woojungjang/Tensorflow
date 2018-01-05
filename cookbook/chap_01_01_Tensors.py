# Tensors(cookbook p29, chap_01_01_Tensors.py)
#----------------------------------
#
# This function introduces various ways to create
# tensors in TensorFlow


# zeros(
#     shape,
#     dtype=tf.float32, (안넣어주면 default가 float32)
#     name=None
# )

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph() #이전에 했던작업 깔끔하게 클리어 파이썬 tf에 해당하는?

# Introduce tensorflow as tf

# Get graph handle
sess = tf.Session()
#0으로 채워진 텐서 생성(1행20열)
my_tensor = tf.zeros([1,20])
print( sess.run(my_tensor))
# Declare a variable
my_var = tf.Variable(tf.zeros([1,20]))

# Different kinds of variables
row_dim = 2
col_dim = 3 

# Zero initialized variable(2행3열을 0으로 초기화)
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))

# One initialized variable
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))

# shaped like other variable
sess.run(zero_var.initializer)
sess.run(ones_var.initializer)
zero_similar = tf.Variable(tf.zeros_like(zero_var)) #zero var를 이용하여 기존텐서를 초기화하기, 기존에 이미 파생되어있는 변수가지고 만들어내기
ones_similar = tf.Variable(tf.ones_like(ones_var))

#session영역에 개별변수를 초기화할때 initializer를 사용하면 된다.
sess.run(ones_similar.initializer)
sess.run(zero_similar.initializer)

# Fill shape with a constant
fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))#[2,3]을 -1로 채운다

# Create a variable from a constant
const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
# This can also be used to fill an array:
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))

# Sequence generation#0.0에서 시작해서 1.0사이에서 3개를 만들어라.(2등분) num=5 [0, 0.25, 0.5, 0.75, 1]
linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end

sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end
#[6, 9, 12] seq
# Random Numbers

# Random Normal#평균이 0.0이고 표준편차가 1.0인 랜덤한 값을 2행3열 생성한다.
rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

# Add summaries to tensorboard
merged = tf.summary.merge_all()

# Initialize graph writer:

writer = tf.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

# Initialize operation
initialize_op = tf.global_variables_initializer()

# Run initialization of variable
sess.run(initialize_op)