import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
# 붓꽃의 CSV 데이터 읽어 들이기 --- (※1)
csv = pd.read_csv('iris.csv')
# 필요한 열 추출하기 --- (※2)
csv_data =csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
csv_label = csv["Name"]
# 학습 전용 데이터와 테스트 전용 데이터로 나누기 --- (※3)
x_train, x_test, y_train, y_test = \
train_test_split(csv_data, csv_label)
# 데이터 학습시키고 예측하기 --- (※4)
clf = svm.SVC()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)

# 정답률 구하기 --- (※5)
ac_score = metrics.accuracy_score(y_test, prediction)
print("정답률 =", ac_score)
cl_report = metrics.classification_report( y_test, prediction)
print("\n리포트 =", cl_report)