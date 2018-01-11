from bs4 import BeautifulSoup
from konlpy.tag import Twitter
fp = open("president_address.txt", "r", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")
text = soup.string
print( text )
print('------------------------')

# 텍스트를 한 줄씩 처리하기 --- (※2)
twitter = Twitter()
word_dic = {}
lines = text.split("\n")
for line in lines: # 한 줄 단위로 형태소 분석
    malist = twitter.pos(line)
    for word in malist:
        if word[1] == "Noun": # 명사 확인하기 --- (※3)
            if not (word[0] in word_dic):
                word_dic[word[0]] = 0
            word_dic[word[0]] += 1 # 카운트하기
# 많이 사용된 명사 출력하기 --- (※4)
keys = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)
for word, count in keys[:50]:
    print("{0}({1}) ".format(word, count), end="")
print()