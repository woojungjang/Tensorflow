import pandas as pd
from sklearn import svm, metrics
# XOR 연산
xor_input = [
[0, 0, 0],
[0, 1, 1],
[1, 0, 1],
[1, 1, 0]
]
# 입력을 학습 전용 데이터와 테스트 전용 데이터로 분류하기 --- (※1)
# 데이터 프레임으로 만든다.
xor_df = pd.DataFrame(xor_input)
# 슬라이싱을 이용하여 데이터와 레이블을 분리한다.
x = xor_df.ix[:,0:1] # 데이터 
y = xor_df.ix[:,2] # 레이블
# 데이터 학습과 예측하기 --- (※2)
# svm 알고리즘을 이용하여 머신 러닝 수행
clf = svm.SVC()
# fit(학습용_데이터, 레이블_데이터) 함수 : 데이터를 학습시킨다.
clf.fit(x, y)
# predict(예측하고자_하는_데이터)
pre = clf.predict(x)
# 정답률 구하기 --- (※3)
# accuracy_score 함수를 이용하면 정답률을 쉽게 구해준다.
# accuracy_score(정답, 예측_결과_배열)
ac_score = metrics.accuracy_score(y, pre)
print("정답률 =", ac_score)
