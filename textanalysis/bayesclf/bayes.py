import math, sys
from konlpy.tag import Twitter
class BayesianFilter:
    """ 베이지안 필터 """
    def __init__(self):
        self.words = set() # 출현한 단어 기록('진행', '확인', '배송', '일정')
        self.word_dict = {} # 카테고리마다의 출현 횟수 기록
        self.category_dict = {} # 카테고리 출현 횟수 기록

# 		'''
# 		변수 word_dict 내용 예시
# 		{
# 			'광고': 
# 				{'쿠폰': 1, '계' '한정': 1, '파격': 1, '할인': 1}, 
# 			'중요': 
# 				{'진행': 1, '등록': 1, '계약': 1, '확인': 1}
# 		}
# 		'''
# 		'''
# 		변수 : category_dict
# 		{'광고': 5, '중요': 5}
# 		'''
        
    # 형태소를 분석하여 list에 저장하고 반환해준다. --- (※1)
    def mysplit(self, text): # 형태소로 나누어 리스트에 담아 준다.
        # 예시 : "파격 세일 - 오늘까지만 30% 할인"
        results = []
        twitter = Twitter()
        # 단어의 기본형 사용
        malist = twitter.pos(text, norm=True, stem=True)
        for word in malist:
            # 어미/조사/구두점 등은 대상에서 제외 
            if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                results.append(word[0])
        return results
    
    # 단어와 카테고리의 출현 횟수 세기 --- (※2)
    def inc_word(self, word, category):
        # 단어를 카테고리에 추가하기
        if not category in self.word_dict: # 사전에 카테고리가 없으면...
            self.word_dict[category] = {} # 카테고리를 사전에 넣고...
            
		# 카테고리를 저장하고 있는 사전에 해당 카테고리가 없으면...
        if not word in self.word_dict[category]:
            self.word_dict[category][word] = 0 # 추가하고...
        self.word_dict[category][word] += 1 # 카테고리 갯수를 +1 하기
        self.words.add(word)
        
    def inc_category(self, category):
        # 카테고리 계산하기
        if not category in self.category_dict:
            self.category_dict[category] = 0
        self.category_dict[category] += 1
    
    # 텍스트 학습하기 --- (※3)
    def fit(self, text, category):
        # 예시 : "파격 세일 - 오늘까지만 30% 할인", "광고"
        """ 텍스트 학습 """
        word_list = self.mysplit(text) # 클래스 내부
        for word in word_list:
            self.inc_word(word, category)
        self.inc_category(category)
    
    
    def showInfo(self):
        print(self.words)
        print(self.word_dict)
        print(self.category_dict)
    
    # 단어 리스트에 점수 매기기--- (※4)
    def score(self, words, category):
        score = math.log(self.category_prob(category)) #downflow를 막기위해서 log를 씀
        for word in words:
            score += math.log(self.word_prob(word, category))
        return score
    
    # 예측하기 --- (※5)
    def predict(self, text): # 
        best_category = None
        max_score = -sys.maxsize 
        words = self.mysplit(text)
        score_list = []
        for category in self.category_dict.keys():
			# 이 단어들과 카테고리에 대하여 ...
            score = self.score(words, category)
            score_list.append((category, score))
            if score > max_score:
                max_score = score
                best_category = category
        return best_category, score_list
    
    # 카테고리 내부의 단어 출현 횟수 구하기
    def get_word_count(self, word, category):
        if word in self.word_dict[category]:
            return self.word_dict[category][word]
        else:
            return 0
        
    # 카테고리 계산
    def category_prob(self, category):
		# 해당 카테고리 내의 모든 값의 합
        sum_categories = sum(self.category_dict.values())
        category_v = self.category_dict[category]
        return category_v / sum_categories
        
    # 카테고리 내부의 단어 출현 비율 계산 --- (※6)
    def word_prob(self, word, category):
        # 사전에 없는 단어가 들어오면 확률이 0이다.  #처음들어온단어 확률=0 광고성메일인지 비광고성메일인지 파악
        n = self.get_word_count(word, category) + 1 # ---(※6a)
        d = sum(self.word_dict[category].values()) + len(self.words)
        return n / d
    
    #필터단어수제한적인모델