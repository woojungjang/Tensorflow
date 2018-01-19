# Text Helper Functions
#---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import string
import os
import urllib.request
import io
import tarfile
import collections
import numpy as np
import requests
import gzip

# Normalize text 
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts] #소문자

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts] #구두점삭제

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts] #숫자없애기

    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts] #불용어 없애기

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts] #띄어쓰기 없애기
    
    return(texts)


# Build dictionary of words 
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]
    
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    
    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    
    return(word_dict)
    

# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split():
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)
    

# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # 일괄 작업 데이터 채우기
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # 시작할 문장을 임의로 선택한다.
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        
        # 탐색할 연속 범위를 생성한다.
        window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        
        # 단어의 중심 단어를 표시한다.
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        
        # 범위에서 중심 단어를 추출하고, 범위에 해당하는 단어 쌍을 생성한다.
        if method=='skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
            
            # 커다란 튜플 리스트 생성
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
            batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method=='cbow':
            batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x,y in zip(window_sequences, label_indices)]
            
            # 2 * window_size의 범위에 있는 데이터만 처리한다.
            batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x)==2*window_size]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method=='doc2vec':
            # doc2vec의 경우 왼쪽 범위만으로 대상 단어를 예측하도록 한다.
            batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]

            # 문서 색인을 일괄 작업에 추가한다.
            # 마지막으로 일괄 작업 번호를 문서 색인 값으로 사용한다.
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))
            
        # 일괄 작업과 라벨 추출
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
        
    # 마지막 데이터 잘라 내기
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    
    # numpy 배열 형식으로 변환한다.
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return(batch_data, label_data)
    
    
###################################################################################################
# 함수 load_movie_data : 파일을 읽어 들여서 데이터를 만들어 준다.
# Load the movie review data
# Check if data was downloaded, otherwise download it and save for future use
###################################################################################################
def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos') # pos 좋은평, neg나쁜평, 
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg') # 파일객체로 만들어줌

    # Check if files are already downloaded
    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open('temp_movie_review_temp.tar.gz', 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        # Extract tar.gz file into temp folder
        tar = tarfile.open('temp_movie_review_temp.tar.gz', "r:gz") # tar zip파일과 동일
        tar.extractall(path='temp')
        tar.close()

    pos_data = []
    with open(pos_file, 'r', encoding='latin-1') as f:
        for line in f:
            pos_data.append(line.encode('ascii',errors='ignore').decode())
    f.close()
    pos_data = [x.rstrip() for x in pos_data]

    neg_data = []
    with open(neg_file, 'r', encoding='latin-1') as f:
        for line in f:
            neg_data.append(line.encode('ascii',errors='ignore').decode())
    f.close()
    neg_data = [x.rstrip() for x in neg_data]
    
    texts = pos_data + neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    
    return(texts, target)