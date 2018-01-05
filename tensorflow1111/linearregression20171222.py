import tensorflow as tf
import numpy as np
 
# 최적의 w와 b를 기계 학습 시켜보세요.

# 비용 함수와 가중치의 변화율을 그래프로 표현하세요. 
 
# 단계 1 : Build graph using TF operations
# x : 입력 데이터, y : 출력될 데이터
# x = [1.0, 2.0, 3.0]
x = np.arange( 1.0, 3.1, 0.5 )
y = [4.9, 6.6, 8.2, 9.3, 10.9]
# 정답 y = [5, 6.5, 8, 9.5, 11]
 
print( 'x = ', x )
print( 'y = ', y )
 
# 가중치와 바이어스의 초기 값을 설정한다.
w = tf.Variable(0.1)
b = tf.Variable(0.1)
 
# 가설을 만든다.
H = w * x + b
 
# cost 함수를 작성한다.
diff = tf.square( H - y )
cost = tf.reduce_mean( diff )
 
# Minimize : 경사 하강법에 의한 최소화 작업
learn_rate = 1e-3
optimizer = tf.train.GradientDescentOptimizer( learning_rate = learn_rate )
train = optimizer.minimize( cost )
 
# 단계 2,3 : Run/update graph and get results
# Launch the graph in a session.
sess = tf.Session()
 
# Initializes global variables in the graph.
sess.run( tf.global_variables_initializer() )
 
for step in range(10000):
    sess.run( train )
    print('step : %d, cost : %f, w : %f, b : %f' % \
        ( step, sess.run(cost), sess.run(w), sess.run(b)))
    
    # w:2.9821
    # b:2.0071
import matplotlib.pyplot as plt    
 
cost_list = []
weight_list = []
for step in range(10000):
    sess.run( train )
    cost_list.append(sess.run(cost))
    weight_list.append(sess.run(w))
    
    import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict


xs = []
ys = []

# 일교차 xs (1:날짜, 4:일교차)

xdata = []
with open('C:/weather_seoul_2012_2015.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print("일교차 : " +row[4])
        tmp = []
        tmp.append(row[1])
        tmp.append(row[4])
        xs.append(tmp)
# print(xs)

dxs = {}
dxs = dict(xs)
#print(dxs)


# 처방횟수 ys
with open('C:/N06AB_2012_2015_RE.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print("처방횟수 = " + row[7])
        ys.append(row[7])

x = []
x = [[x,ys.count(x)] for x in set(ys)]
ys =sorted((x), reverse=False)
# print(ys)

dys = {}
dys = dict(ys)
#print(dys)


def FindRecuperate(dys, e1):
    for key in dys.keys():
        if key == e1:
            return dys[key]

    return 0


# RECUPERATE_DATE
# xs 에 해당 ys 값을 추가한다

data = []
xs = []
ys = []
for key in dxs.keys():
    d3 = FindRecuperate(dys, key)
    print(d3)

    if d3:
        tmp = []
        # tmp.append(key)
        # tmp.append(dxs[key])
        xs.append( float(dxs[key]) )
        ys.append(float(d3))
print(xs)
print(ys)


# Weight(가중치) 예상치
# 2 ~ 4 사이의 랜덤 값으로 제한
# W = tf.Variable(tf.random_uniform([1], 2, 4))
W = tf.Variable(tf.random_uniform([1], 2, 4))
# W = tf.Variable(4.0)

# bias 예상치
# -2 ~ -2 사이의 랜덤 값으로 제한
# b = tf.Variable(tf.random_uniform([1], -2, 2))
b = tf.Variable(tf.random_uniform([1], -2, 2))
# b = tf.Variable(2.0)

# 가설 함수
hypothesis = W * xs + b




# 비용 함수
cost = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(ys, hypothesis), 2)))


# 최적화 객체
# 학습률을 0.1 로 지정했다.
# optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.GradientDescentOptimizer(0.000001)

# 비용 함수(cost)를 최소화(minimize) 하기 위한 W, b 를 구하도록 최적화 객체에게 지시한다.
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# 10000 번을 반복하면서 최적의 W, b 를 구한다.
for step in range(1001):
    sess.run(train)

    if step % 10 == 0:
        #plt.plot(xs, ys, 'ro')
        #plt.plot(xs, sess.run(hypothesis), 'b')
        #plt.xlim(0.0, 3.0)
        #plt.ylim(0.1, 6.0)
        #plt.title("step: {} / cost: {}\nW: {} / b: {}".format(step, sess.run(cost), sess.run(W), sess.run(b)))
        #plt.show()

        print(sess.run(hypothesis))
        print("step: {} / cost: {}\nW: {} / b: {}".format(step, sess.run(cost), sess.run(W), sess.run(b)))



