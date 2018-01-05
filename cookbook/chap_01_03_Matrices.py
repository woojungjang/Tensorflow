#chap_01_03_Matrices.py  
#Matrices and Matrix Operations(쿡북37)
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Declaring matrices
sess = tf.Session()

# Declaring matrices

# Identity matrix
#3행3열의 단위행렬생성하기
identity_matrix = tf.diag([1.0,1.0,1.0])
print(sess.run(identity_matrix))

# 2x3 random norm matrix
A = tf.truncated_normal([2,3])
print(sess.run(A))

# 2x3 constant matrix
#2행3열을 모두 5로 채우기
B = tf.fill([2,3], 5.0)
print(sess.run(B))

# 3x2 random uniform matrix #균등분포를 따르는 난수값으로 텐서를 생성한다.
C = tf.random_uniform([3,2])
print(sess.run(C))
print(sess.run(C)) # Note that we are reinitializing, hence the new random variabels

# Create matrix from np array #convert_to_tensor 배열을 텐서객체로 만들어주는 함수 nd.array =>텐서타입 np.array
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))

# Matrix addition/subtraction
print(sess.run(A+B))
print(sess.run(B-B))

# Matrix Multiplication
print(sess.run(tf.matmul(B, identity_matrix))) #매트릭스연산

# Matrix Transpose #행렬전치
print(sess.run(tf.transpose(C))) # Again, new random variables

# Matrix Determinant #행렬식구하기 역행렬 =>행렬식ad-bc
print(sess.run(tf.matrix_determinant(D)))

E = tf.convert_to_tensor(np.array([[4.,2.],[-3.,5.]]))
print(sess.run(tf.matrix_determinant(E)))

# Matrix Inverse
print(sess.run(tf.matrix_inverse(D)))

# Cholesky Decomposition
print(sess.run(tf.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors #고유치와 고유백터
print(sess.run(tf.self_adjoint_eig(D)))