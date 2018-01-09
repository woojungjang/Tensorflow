import tensorflow as tf
import numpy as np
import os
import requests
import string
import collections

from nltk.corpus import stopwords
###################################################################################################
# 변수 선언 Declare model parameters
###################################################################################################
batch_size = 100 # 일괄 작업 크기(한번에 임베딩할 크기)
embedding_size = 200 # 각 단어의 임베딩 크기(길이가 200인 벡터)
# 빈도가 높은 10000개만 임베딩 대상으로 하겠다.(나머지는 알수 없음으로 분류)
vocabulary_size = 10000 

generations = 100000 # 반복 학습 횟수
print_loss_every = 2000 # 2000번 마다 비용 함수의 결과 값 출력하기 

num_sampled = int(batch_size/2)    # Number of negative examples to sample.

# How many words to consider left and right.
# 스킵 그램의 범위(숫자 2는 대상 단어의 양쪽 두 단어를 살펴 보겠다.)
window_size = 2

# 파이썬 nltk 패키지의 불용어를 설정한다.
stops = stopwords.words('english')
# print( len(stops) ) # 179

# 검정 단어(관심을 가지고 지켜 봐야 할 단어)
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']

# 단어 임베딩 결과를 알아볼 수 있게 일반적인 몇 가지 단어에 대한 근접한 이웃 단어들을 5000회 마다 출력하겠다.
print_valid_every = 5000

# Later we will have to transform these into indices
###################################################################################################
# 함수 load_movie_data : 파일을 읽어 들여서 데이터를 만들어 준다.
###################################################################################################
def load_movie_data():
    save_folder_name = 'temp'
    
    # os.path.join : 디렉토리 이름과 파일 이름을 연결하여 전체 경로를 만들어 준다.
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

    pos_data = []
    with open(pos_file, 'r', encoding='latin-1') as f: # 파일 열고...
        for line in f:
            pos_data.append(line.encode('ascii',errors='ignore').decode())
    f.close() # 파일 닫기
    
    # rstrip() 함수 : 오른쪽 공백 제거하기
    pos_data = [x.rstrip() for x in pos_data]
#     print(pos_data)

    neg_data = []    
    with open(neg_file, 'r', encoding='latin-1') as f: # 파일 열고...
        for line in f:
            # print(line.encode('ascii',errors='ignore').decode())
            neg_data.append(line.encode('ascii',errors='ignore').decode())
    f.close() # 파일 닫기
    
    neg_data = [x.rstrip() for x in neg_data]
    
    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] * len(neg_data)
#     print(len(target)) # 10662

    return(texts, target)
###################################################################################################
# 함수 normalize_text : 문서에 대한 정규화 함수
###################################################################################################
def normalize_text(texts, stops):
    texts = [x.lower() for x in texts] # 소문자로 바꾸고...

    # 다음 단어들은 모두 없애 버린다
    # print(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # 숫자들도 모두 없애 버린다.
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # 불용어(stopwords)를 제거한다.
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]

    # 여분의 공백들은 모두 제거한다.
    texts = [' '.join(x.split()) for x in texts]
    
    # print( texts )
    return(texts)
###################################################################################################
# 함수 build_dictionary : 어휘 사전을 만들어 주는 함수
###################################################################################################
def build_dictionary(sentences, vocabulary_size):
    # sentences : 사전을 만들 단어들
    # vocabulary_size : 사용 빈도 순위 1위부터 (vocabulary_size-1)위 까지만...
    # 빈도가 높은 vocabulary_size(10000)까지만 임베딩 대상으로 하겠다.(나머지는 알수 없음으로 분류) 
    # Turn sentences (list of strings) into lists of words
    
    split_sentences = [s.split() for s in sentences]
    # print(split_sentences) # 모든 단어(음절)들의 집합
    
    words = [x for sublist in split_sentences for x in sublist]
    # print(words) # 1차원으로 변환된 결과
    
    # Initialize list of [word, word_count] for each word, starting with unknown
    # 각 단어의 [[단어, 출현횟수]] 리스트를 초기화
    # 임계 값을 넘지 못하는 드물게 출현하는 단어들에 대한 처리
    # 나머지 단어들은 알수 없음으로 처리
    count = [['RARE', -1]]    
    
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    # print(count) # 각 단어들에 대한 빈도 수를 저장하고 있다.
    
    # 사전 정의
    word_dict = {}
    
    # 각각의 단어에 대하여 키와 값을 이용하여 사전에 추가한다.
    for word, word_count in count:
#         print( '키 : ', word, ', 값 :', word_count )
#         word_dict[word] = len(word_dict)
        word_dict[word] = word_count
    
    # word_dict 예시
    # {'that':3, 'the':4, 'RARE':0, 'or':1, 'to':2}
    
    return(word_dict)
###################################################################################################
# 함수 text_to_numbers : 문장 리스트를 단어 색인으로 바꿔주는 함수
###################################################################################################
def text_to_numbers(sentences, word_dict):
    # sentences : 단어들 목록
    # word_dict : 해당 단어들 각각이 key로 등록되어 있는 사전
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split(' '):
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0 # RARE 단어는 0으로 치환
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)
###################################################################################################
# 함수 generate_batch_data : 스킵 그램을 일괄 작업해주는 함수
# 학습시 반복문에서 계속 호출이 된다.
###################################################################################################
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence = np.random.choice(sentences)
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        
        # Pull out center word of interest for each window and create a tuple for each window
        if method=='skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
        elif method=='cbow':
            batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x,y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x_, y) for x,y in batch_and_labels for x_ in x]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))
            
        # extract batch and labels
        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
        
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return(batch_data, label_data)
###################################################################################################
texts, target = load_movie_data()

texts = normalize_text(texts, stops)

# texts는 양이 너무 많아서 출력이 되지 않는 것처럼 보인다.
print('texts\n', texts)
print('\ntarget\n', target)

# Texts must contain at least 3 words
# 단어 사이의 관계가 확실히 reviews에 대한 정보라는 것을 보장 받기 위해서
# 길이는 3이상이어야 한다.
# 예를 들어서, 'hi hello'는 리뷰가 아니다. 라고 보는 것이다.
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# Build our data set and dictionaries
word_dictionary = build_dictionary(texts, vocabulary_size)

# word_dictionary_rev : 인덱스와 키를 바꾼 사전 구조
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))

# text_data : 해당 단어들에 대한 색인 정보를 담고 있다.
text_data = text_to_numbers(texts, word_dictionary)

# valid_examples : 검정하고자 하는 단어들의 색인 값을 저장하고 있다.
valid_examples = [word_dictionary[x] for x in valid_words]

# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
print(embeddings.shape) # shape(10000, 200)

# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                               stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding:
embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
                                     
# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run the skip gram model.
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
    feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)

    # Return the loss
    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print("Loss at step {} : {}".format(i+1, loss_val))
      
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
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)
###################################################################################################            