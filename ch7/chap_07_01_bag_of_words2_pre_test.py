import tensorflow as tf
import os
import numpy as np
import csv
import string
from tensorflow.contrib import learn

# os.path.join : 디렉토리 이름과 파일 이름을 연결하여 전체 경로를 만들어 준다.
save_file_name = os.path.join('temp','abcd.csv')

text_data = []
with open(save_file_name, 'r') as temp_output_file: # 파일을 열고 ...
    reader = csv.reader(temp_output_file)
    for row in reader: # 각 라인마다..
        # row는 1줄을 의미하는 데 콤마 1개로 분리되어 있다.
        # 콤마 왼쪽이 y(정답 laebl), 오른쪽이 x(입력할 데이터)
        # 예시 : ham,Ok lar12aaa 
        text_data.append(row) # 리스트에 추가한다.
   
texts = []
target = []
for item in text_data :
    if len(item) == 2 :
        texts.append( item[1] )
        target.append( item[0] )

print('\ntexts')
print(texts)

print('\ntarget befor7un e')
print(target)

# target은 ham과 spam 중 하나가 들어 있는 문자열이다.
# spam은 숫자 1로 ham은 숫자 0으로 변경한다.
target = [1 if x=='spam' else 0 for x in target]

print('\ntarget after')
print(target)

texts = [x.lower() for x in texts]
print('\ntexts')
print(texts)
  
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
print('\ntexts')
print(texts)

texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
print('\ntexts')
print(texts)

# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]
print('\ntexts')
print(texts)

# Plot histogram of text lengths
# text_lengths : 각 행당 단어들의 갯수를 list 구조로 가지고 있다.
text_lengths = [len(x.split()) for x in texts]
print('\ntext_lengths')
print(text_lengths)

su = 3
# su보다 작은 항목들만 필터링한다.
# 각 행마다 su개 미만의 단어로 구성된 행만 취하겠다.
text_lengths = [x for x in text_lengths if x < su]
print('\ntext_lengths')
print(text_lengths)

# Choose max text word length at 25
# sentence_size : 문서 처리에 적용할 단어의 최대 개수
# 즉, 모든 문장들이 해당 길이의 벡터가 되도록 만들어 준다.
sentence_size = 5 #
min_word_freq = 3
 
# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
print('\nvocab_processor')
print(vocab_processor) 
 
# Have to fit transform to get length of unique words.
print(type(vocab_processor.transform(texts)))
aaa = [x for x in vocab_processor.transform(texts)]
print('\naaa')
print(aaa) 

embedding_size = len([x for x in vocab_processor.transform(texts)])
print('\nembedding_size')
print(embedding_size) 

# 학습 셋(80%)과 테스트 셋(20%)으로 분할한다.
train_indices = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))

texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]

target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

print('\ntexts_train', texts_train)
print('\ntexts_test', texts_test)
print('\ntarget_train', target_train)
print('\ntarget_test', target_test)
  
sess = tf.Session()  

diagonal = tf.ones(shape=[embedding_size])
print('diagonal')
print(sess.run(diagonal))

identity_mat = tf.diag(diagonal)
print('identity_mat.shape', identity_mat.shape)
print('\nidentity_mat\n', sess.run(identity_mat))
  
# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Initialize placeholders
# x_data는 한 행당 sentence_size(5) 만큼의 길이를 가지므로
# x_data.shape (5,)이다.
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)
print('x_data.shape', x_data.shape) # x_data.shape (5,)
print('y_target.shape', y_target.shape) # y_target.shape (1, 1)

# Text-Vocab Embedding
# 임베딩 조회 함수를 이용하여 문장의 단어 인덱스를 단위 행렬의 
# 원 핫 인코딩 벡터로 할당한다.
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
print('x_embed.shape', x_embed.shape) # x_embed.shape (5, 3)

# x_embed Tensor("embedding_lookup:0", shape=(5, 3), dtype=float32)
print('x_embed', x_embed)

# x_col_sums Tensor("Sum:0", shape=(3,), dtype=float32)
x_col_sums = tf.reduce_sum(x_embed, 0)

# x_col_sums Tensor("Sum:0", shape=(3,), dtype=float32)
print('x_col_sums', x_col_sums)

# Declare model operations
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
print('x_col_sums_2D.shape', x_col_sums_2D.shape) # x_col_sums_2D.shape (1, 3)

# x_col_sums_2D Tensor("ExpandDims:0", shape=(1, 3), dtype=float32)
print('x_col_sums_2D', x_col_sums_2D)