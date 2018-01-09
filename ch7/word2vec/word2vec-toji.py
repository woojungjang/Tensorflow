import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import word2vec

fp = codecs.open("beatles_yesterday.txt", "r", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")

text = soup.string
 
# 텍스트를 한 줄씩 처리하기 --- (※2)
twitter = Twitter()
results = []
lines = text.split("\r\n")

for line in lines:
    # 형태소 분석하기 --- (※3)
    # 단어의 기본형 사용
    malist = twitter.pos(line, norm=True, stem=True)
    r = []
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외 
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    results.append(rl)
    print(rl)
    
# 파일로 출력하기  --- (※4)
wakati_file = 'yesterday.model'
with open(wakati_file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(results))
    
# Word2Vec 모델 만들기 --- (※5)
# LineSentence 함수로 텍스트 파일을 읽어 들인다.
data = word2vec.LineSentence(wakati_file)
model = word2vec.Word2Vec(data, 
    size=200, window=10, hs=1, min_count=2, sg=1)

# 모델을 저장한다.
model.save("yesterday.model")
print("ok")