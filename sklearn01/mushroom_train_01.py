import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
# 데이터 읽어 들이기--- (※1)
mr = pd.read_csv("mushroom.csv", header=None)
# 데이터 내부의 기호를 숫자로 변환하기--- (※2)
y = []
x = []

for row_index, row in mr.iterrows():
    y.append(row.ix[0]) # 독의 유무(p_독성, e_식용)
    row_data = []
    for v in row.ix[1:]:
        row_data.append(ord(v))
    x.append(row_data)
# 학습 전용과 테스트 전용 데이터로 나누기 --- (※3)
x_train, x_test, y_train, y_test = \
    train_test_split(x, y)
    
# 데이터 학습시키기 --- (※4)
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

# 데이터 예측하기 --- (※5)
predict = clf.predict(x_test)
# 결과 테스트하기 --- (※6)
ac_score = metrics.accuracy_score(y_test, predict)
print("정답률 =", ac_score)
cl_report = metrics.classification_report(y_test, predict)
print("리포트 =\n", cl_report)