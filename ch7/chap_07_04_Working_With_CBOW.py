import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
import text_helpers
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))
###################################################################################################
# 변수 선언 
###################################################################################################
data_folder_name = 'temp' # 파일이 있는 폴더 경로

sess = tf.Session()

batch_size = 500 # 일괄 작업 크기(한번에 임베딩할 크기)
embedding_size = 200 # 각 단어의 임베딩 크기(길이가 200인 벡터)

# 빈도가 높은 2000개만 임베딩 대상으로 하겠다.(나머지는 알수 없음으로 분류)
vocabulary_size = 2000 

print_loss_every = 100 # 100번 마다 비용 함수의 결과 값 출력하기 

generations = 50000 # 반복 학습 횟수
model_learning_rate = 0.001

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 3       # How many words to consider left and right.

# 파이썬 nltk 패키지의 불용어를 설정한다.
stops = stopwords.words('english')
# print( len(stops) )

# We pick some test words. We are expecting synonyms to appear
# 검정 단어(관심을 가지고 지켜 봐야 할 단어)
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
# Later we will have to transform these into indices

# Add checkpoints to training
save_embeddings_every = 5000

# 단어 임베딩 결과를 알아볼 수 있게 일반적인 몇 가지 단어에 대한 근접한 이웃 단어들을 5000회 마다 출력하겠다.
print_valid_every = 5000
###################################################################################################
# 함수
###################################################################################################
if not os.path.exists(data_folder_name):# 해당 디렉토리가 없으면
    os.makedirs(data_folder_name)# 해당 디렉토리 생성

# Load the movie review data
print('Loading Data') # 파일을 읽어 들여서 데이터를 만들어 준다.
texts, target = text_helpers.load_movie_data()

# 함수 normalize_text : 문서에 대한 정규화 함수
print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)

# Texts must contain at least 3 words
# 3 단어(if len(x.split()) > 2) 이상만 처리 대상으로 간주한다.
# 예를 들어서, 'hi hello'는 리뷰가 아니다. 라고 보는 것이다.
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]    

# 함수 build_dictionary : 어휘 사전을 만들어 주는 함수
print('Creating Dictionary')
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))

# 함수 text_to_numbers : 문장 리스트를 단어 색인으로 바꿔주는 함수
text_data = text_helpers.text_to_numbers(texts, word_dictionary)

# valid_examples : 검정하고자 하는 단어들의 색인 값을 저장하고 있다.
valid_examples = [word_dictionary[x] for x in valid_words]    

print('Creating Model')
# Define Embeddings:
# 최적화하려는 임베딩을 초기화한다.
# 빈도가 높은 vocabulary_size(2,000) 개만 embedding_size(200) 만큼 만들겠다.
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# 모델의 데이터 플레이스 홀더를 선언한다.
x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 임베딩 처리 방법을 정의한다.
# batch_size = 500, embedding_size = 200
embed = tf.zeros([batch_size, embedding_size])

# cbow 모델은 맥락 범위에 걸친 임베딩 값을 더하므로
# 루프를 통하여 범위에 해당하는 모든 임베딩 값을 합산한다.
for element in range(2 * window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                               stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Get loss from prediction
# 모델이 출력하는 분류 값이 너무 희소해서 softmax 함수로는 수렴이 안된다.
# 텐서 플로우 내의 nce 비용 함수를 사용한다.
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
                                     
# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)

# 임베딩 동작 과정을 살펴 보기 위해 코사인 유사도를 사용하여
# 검증 단어 데이터 셋의 가장 가까운 단어를 출력한다.
# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# 임베딩 변수를 저장하기 위해 Saver 메소드를 호출한다.
saver = tf.train.Saver({"embeddings": embeddings})

#Add variable initializer.
init = tf.global_variables_initializer()
sess.run(init)

# Filter out sentences that aren't long enough:
text_data = [x for x in text_data if len(x)>=(2*window_size+1)]

# Run the CBOW model.
print('Starting Training')
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size,
                                                                  window_size, method='cbow')
    feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)

    # Return the loss
    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))
      
    # Validation: Print some random words and top 5 related words
    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5 # number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},' .format(log_str, close_word)
            print(log_str)
            
    # Save dictionary + embeddings
    if (i+1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name,'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)
        
        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(),data_folder_name,'cbow_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))