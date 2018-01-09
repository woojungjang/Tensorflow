import numpy as np
import collections

texts = ['aaa bbb ccc ddd aaa bbb ccc aaa bbb aaa']

# Build dictionary of words
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
#     print(split_sentences) # 모든 단어(음절)들의 집합
    
    words = [x for sublist in split_sentences for x in sublist]
#     print(words) # 1차원으로 변환된 결과
    
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]
    
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
#     print(count) # 각 단어들에 대한 빈도 수를 저장하고 있다.
    
    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
#         print( '키 : ', word, ', 값 :', word_count )
#         word_dict[word] = len(word_dict)
        word_dict[word] = word_count
    
    return(word_dict)
    

# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split(' '):
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)

vocabulary_size = 10 # 3으로 변경해보세요.
# Build our data set and dictionaries
word_dictionary = build_dictionary(texts, vocabulary_size)
print('\nword_dictionary', word_dictionary)

word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
print('\nword_dictionary_rev', word_dictionary_rev)

text_data = text_to_numbers(texts, word_dictionary)
print('\ntext_data', text_data)

valid_words = ['aaa', 'bbb']

# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]
print('\nvalid_examples', valid_examples)

# Generate data randomly (N words behind, target, N words ahead)
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

    
    
    