import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import string
import os
import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
###################################################################################################
# 변수 선언 
###################################################################################################
max_features = 1000
###################################################################################################
# 함수 get_tokenizer : tokenizer를 수행해주는 함수 구현
###################################################################################################
# TF-IDF 어휘 처리 함수를 사용하려면 문장 분할 방식을 알려 줘야 한다.
# nltk 패키지에 문장을 단어들로 분할해주는 tokenizer가 내장되어 있다.
def get_tokenizer(text):
    # word_tokenize : 텍스트를 공백으로 분할 시켜 list로 반환해준다.
    words = nltk.word_tokenize(text)
#     print(words)
    return words
###################################################################################################
# os.path.join : 디렉토리 이름과 파일 이름을 연결하여 전체 경로를 만들어 준다.
save_file_name = os.path.join('temp','temp_spam_data.csv')
# save_file_name = os.path.join('temp','bbb.csv')
# save_file_name = os.path.join('temp','aaa.csv')

texts = [] # 입력 데이터
target = [] # 정답 데이터
 
with open(save_file_name, 'r') as temp_output_file:
    reader = csv.reader(temp_output_file)
    for row in reader:
        if len(row) == 2 : # 중간에 비어 있는 행이 발견되어 if 구문으로 처리
            texts.append(row[1])
            target.append(row[0])
            
# target은 ham과 spam 중 하나가 들어 있는 문자열이다.
# spam은 숫자 1로 ham은 숫자 0으로 변경한다.            
target = [1. if x=='spam' else 0. for x in target]

texts = [x.lower() for x in texts]
# 문장 부호(구두점) 제거
# print(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]

print('\ntexts', texts)
print('\ntarget', target)

# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=get_tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

print(type(sparse_tfidf_texts)) # <class 'scipy.sparse.csr.csr_matrix'>
print('\nsparse_tfidf_texts.shape :', sparse_tfidf_texts.shape) # (3, 6)
print(sparse_tfidf_texts)
print('\n전체 행수 :', sparse_tfidf_texts.shape[0])
            
# 전체 행수를 80:20으로 학습용셋과 테스트셋으로 분리한다.
# sparse_tfidf_texts.shape[0] : 샘플의 갯수(엑셀 파일의 행수)를 의미한다.
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))

texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]

target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
 
print('\ntexts_train\n', texts_train)
print('\ntexts_test\n', texts_test)

print('\ntarget_train\n', target_train)
print('\ntarget_test\n', target_test) 
 
# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[max_features,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
 
# Initialize placeholders
x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
 
# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(x_data, A), b)
 
# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
 
# 예측 함수 정의
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)

# 정확도 함수 정의
accuracy = tf.reduce_mean(predictions_correct)
 
# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)
 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)    
            
# Start Logistic Regression
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []

batch_size = 200

for i in range(10000):
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
     
    # Only record loss and accuracy every 100 generations
    if (i+1)%100==0:
        i_data.append(i+1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)
         
        test_loss_temp = sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)
         
        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)
     
        test_acc_temp = sess.run(accuracy, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)
    if (i+1)%500==0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
#         print(texts)
 
# 훈련용 셋과 테스트 셋의 비용 함수 그래프
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()
 
# 훈련용 셋과 테스트 셋의 정확도 그래프
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()           
            
print('끝')