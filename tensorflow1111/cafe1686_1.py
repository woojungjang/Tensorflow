

# csvReaderEx03.py
# csv 파일에서 데이터 읽어 오기
import tensorflow as tf
import numpy as np
 
 
# loadtxt : 데이터를 튜플 형식으로 반환해준다.
data = np.loadtxt('./sample_data.csv', dtype=np.float32, delimiter=',')
data = data[:, 1:]
# print(data)
#print(data.shape) 
# data.shape는 tuple 자료형인데, 인덱싱이 가능하다.


table_row = data.shape[0]
print(table_row)
table_col = data.shape[1]
print(table_col)
 
column = table_col - 1 # 입력 데이터의 컬럼 갯수
 

testing_row = 10 # 테스트 용 데이터 셋 개수
training_row = table_row - testing_row # 훈련용 데이터 셋 개수
 
print( 'table_row : %d, training_row : %d' % (table_row , training_row))
 
def normalize(input):
    max = np.max(input,axis=0)
    min = np.min(input,axis=0) 
    out = (input - min)/(max-min)  
    # print (min)
    # print (max)
    return out
 
def rev_normalize(somedata, alist) :
    result = np.min( alist ) + somedata * ( np.max(alist) - np.min(alist) )
    return result
 
x_train = data[ 0:training_row, 0:column ]
y_train = data[ 0:training_row, column:(column+1) ]
 
x_train = normalize(x_train)
y_train = normalize(y_train)
 
x_test  = data[training_row:, 0:column ]
y_test  = data[training_row:, column:(column+1) ]
 
y_test_origin = y_test
 
# print (y_test.shape)
x_test = normalize(x_test)
y_test = normalize(y_test)
 
X = tf.placeholder(tf.float32,shape=[None, column])
Y = tf.placeholder(tf.float32,shape=[None, 1])
 
w = tf.Variable(tf.ones([column, 1], tf.float32))
b = tf.Variable(0.0)
 
H = tf.matmul(X,w) + b
 
loss = tf.square(Y - H)#오류
loss = tf.reduce_mean(loss)#모든 샘플의 오류 평균
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-1)#학습기
train = optimizer.minimize(loss)#학습
 
sess=tf.Session()
sess.run(tf.global_variables_initializer())#변수 초기화
 
epoch =10000
for iter in range(epoch):
    t_,w_,l,h = sess.run([train,w,loss,H], 
                         feed_dict={X:x_train,Y:y_train } )
    if iter%(epoch/10)==0:
        print('iter:%d, loss:%f ' %(iter,l))#학습
 
h=sess.run(H, feed_dict={X:x_test})
print('Input\n', x_test)
print('-------------------------------------')
print('why',sess.run(w))
 
def dataSum():
    # 데이터를 일목 요연하게 보기 위하여 배열들을 합쳐 주는 함수이다.
    totallist = []  # 전체 목록을 담을 리스트
    for i in range(len(y_test)):  # 열의 갯수 만큼 반복
        sublist = []
        sublist.append( y_test[i][0] )
        sublist.append( h[i][0] )
        sublist.append( y_test_origin[i][0] )
        sublist.append(rev_normalize(h, y_test_origin)[i][0])
        totallist.append(sublist)
 
    return totallist
 
print('-------------------------------------')
print('-------------------------------------')
print('\n실제 가격(정규화 데이터)\n', y_test)
print('-------------------------------------')
print('\n학습 결과(정규화 데이터)\n',h)
print('-------------------------------------')
print('-------------------------------------')
print('\n실제 가격\n', y_test_origin)
print('-------------------------------------')
imsi = rev_normalize(h, y_test_origin)
print('\n학습 결과\n', imsi )
print('-------------------------------------')
print('-------------------------------------')
temp = dataSum( )
print( temp )