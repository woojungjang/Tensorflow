from gensim.models import word2vec

data = word2vec.Text8Corpus("president.wakati") 

model = word2vec.Word2Vec(data, size=100) #모델만듬

model.save("president.model")

print("president.model 파일 저장됨 ok")
print('president_play.py 실습 요망')