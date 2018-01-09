from gensim.models import word2vec

model = word2vec.Word2Vec.load('yesterday.model')
print( model )

# KeyError: "word 'love' not in vocabulary"
# vocabulary는 어떤 사전을 말하나
print( model.most_similar(positive=['love']) )
