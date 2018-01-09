import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import string
import os
import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# os.path.join : 디렉토리 이름과 파일 이름을 연결하여 전체 경로를 만들어 준다.
# save_file_name = os.path.join('temp','temp_spam_data.csv')
# save_file_name = os.path.join('temp','bbb.csv')
save_file_name = os.path.join('temp','aaa.csv')

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

# TF-IDF 어휘 처리 함수를 사용하려면 문장 분할 방식을 알려 줘야 한다.
# nltk 패키지에 문장을 단어들로 분할해주는 tokenizer가 내장되어 있다.
# tokenizer를 수행해주는 함수 구현
def get_tokenizer(text):
    words = nltk.word_tokenize(text)
    print('words')
    print(words)
    return words
 
max_features = 1000

# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=get_tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

print(type(sparse_tfidf_texts)) # <class 'scipy.sparse.csr.csr_matrix'>
print('\nsparse_tfidf_texts.shape :', sparse_tfidf_texts.shape) # (3, 6) #3행 = nsamples #3행 접근하려면 shape[0]
print(sparse_tfidf_texts)
print('\n전체 행수 :', sparse_tfidf_texts.shape[0])

# 전체 행수를 80:20으로 학습용셋과 테스트셋으로 분리한다.
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)#천건의 0.8 800개 학습용
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))
texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
 
print('\ntexts_train\n', texts_train)
print('\ntexts_test\n', texts_test)

print('\ntarget_train\n', target_train)
print('\ntarget_test\n', target_test)