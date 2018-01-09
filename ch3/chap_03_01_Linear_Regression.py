# chap_03_01_Linear_Regression.py
# 역행렬기법 사용
# Linear Regression: Inverse Matrix Method(쿡북112쪽)
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve linear regression via the matrix inverse.
#
# Given Ax=b, solving for x:
#  x = (t(A) * A)^(-1) * t(A) * b
#  where t(A) is the transpose of A

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create the data
x_vals = np.linspace(0, 10, 100) #0과 10을 끝점으로 100개의 등간격 데이터
y_vals = x_vals + np.random.normal(0, 1, 100)

# Create design matrix
x_vals_column = np.transpose(np.matrix(x_vals)) #shape(100,1)
ones_column = np.transpose(np.matrix(np.repeat(1, 100))) #shape(100,1)

#각 열에 x_vals 값과 숫자 1이 들어 있는 행렬
# A의 shape() => (100,2)
A = np.column_stack((x_vals_column, ones_column))
                        #(100,1)      (100,1)

# Create b matrix
b = np.transpose(np.matrix(y_vals))

# Create tensors
A_tensor = tf.constant(A) #shape(100,2)
b_tensor = tf.constant(b) #shape(100,1)

# Matrix inverse solution
# (2,100) * (100,2) => (2,2)
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor) #식1 
#(2,2)
tA_A_inv = tf.matrix_inverse(tA_A) #식2
# (2,100)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor)) #식3
# (2,100)*(100,1) => (2,1)

solution = tf.matmul(product, b_tensor) #식4
            #(1,2)*(100,1)
solution_eval = sess.run(solution)

# Extract coefficients
slope = solution_eval[0][0] #기울기
y_intercept = solution_eval[1][0] #편향

print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))

# Get best fit line
best_fit = [] # 직선을 그리기 위한 점정보를 저장하는 리스트
for i in x_vals:
  best_fit.append(slope*i+y_intercept)

# Plot the results
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()