'''
Created on 2017. 12. 21.

@author: acorn
'''
import numpy as np
from _operator import matmul

a = np.array([1,2],)

arrTwoDim = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arrTwoDim)
print('\n# 0-1행 1-2열까지 배열')

b = arrTwoDim[:2, 1:3]
print(b)
print()

b=arrTwoDim[:2, 1:3]

result = np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])

      
''''a = [[[a,a][a,a]][[a,a][a,a]][[a,a][a,a]]]
print(a)'''


'''rank:3 shape: '''

arrA = np.array([300, 80])
arrB = np.array([4,3])
a = np.matmul(arrA, arrB)
print(a)

arrX = np.array([300, 80])
arrY = np.array([4, 3])


result = np.matmul(arrX, arrY)
print('result:', result )

newdata = [1 for i in range(2)]

a = [[-1, 3], [2, 6]]
b = [[3, 6], [1, 2]]

result = np.matmul(a, b)

print(result )

t1 = [[[0,0,0], [1,2,3]],
     [[1,1,1], [4,5,6]],
     [[1,1,1], [4,5,6]]]
t2 = [[[0,0,0], [1,2,3]],
     [[1,1,1], [4,5,6]],
     [[1,1,1], [4,5,6]]]
#result = np.matmul(t1, t2)
#print(result)


A = [[15,5],[0, -5]]
B = [[10,-5],[5, 15]]
C = [[2,2],[2,2]]
print('두배')
result1 = np.matmul(A, C)
result1 = np.matmul(result1, B)
print(result)


def read_data(file_name):

    try:

        csv_file = tf.train.string_input_producer([file_name],name='filename_queue')

        textReader = tf.TextLineReader()

        _,line = textReader.read(csv_file)

        year,flight,time = tf.decode_csv(line,record_defaults=[ [1900],[""],[0] ],field_delim=',')    



