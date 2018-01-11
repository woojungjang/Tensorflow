#참고 api https://radimrehurek.com/gensim/apiref.html
#class gensim.models.word2vec.LineSentence(source, max_sentence_length=10000, limit=None)
#글자길이가 만개까지


from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import word2vec
fp = open("president_address.txt", "r", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")
text = soup.string

# 텍스트를 한 줄씩 처리하기 --- (※2)
twitter = Twitter()
results = []
lines = text.split("\n")
for line in lines:
    # 형태소 분석하기 --- (※3)
    # 동사와 형용사는 단어의 기본형만 사용하도록 하였다.
    malist = twitter.pos(line, norm=True, stem=True)

    r = []
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    results.append(rl)
    print(rl) 
# 파일로 출력하기 --- (※4)
file = 'president_result.txt'
with open(file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(results))
# Word2Vec 모델 만들기 --- (※5)
# LineSentence 함수로 텍스트 파일을 읽어 들인다.
data = word2vec.LineSentence( file )

# Word2Vec 메소드에 매개 변수 형태로 넣어 주면 모델이 만들어 진다.
model = word2vec.Word2Vec(data,
    size=200, window=10, hs=1, min_count=2, sg=1)

# save() 메소드로 모델을 저장한다.
model.save("president.model")

print("ok finished")


