#http://scikit-learn.org/stable/index.html
#sklearn.svm.SVC
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

from sklearn import svm, metrics
import random, re
# 붓꽃의 CSV 데이터 읽어 들이기 --- (※1)
csv = []
with open('iris.csv', 'r', encoding='utf-8') as fp:
    # 한 줄씩 읽어 들이기
    for line in fp:
        line = line.strip() # 줄바꿈 제거
        cols = line.split(',') # 쉼표로 자르기
        # 문자열 데이터를 숫자로 변환하기
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        cols = list(map(fn, cols))
        csv.append(cols)
# 가장 앞 줄의 헤더 제거
del csv[0]
# 데이터 셔플하기(섞기) --- (※2)
random.shuffle(csv)
# 학습 전용 데이터와 테스트 전용 데이터 분할하기(2:1 비율) --- (※3)
total_len = len(csv)
train_len = int(total_len * 2 / 3) # 100개와 50개
x_train = []
y_train = []
x_test = []
y_test = []
for i in range(total_len):
    data = csv[i][0:4]
    label = csv[i][4] # 품종
    if i < train_len:
        x_train.append(data)
        y_train.append(label)
    else:
        x_test.append(data)
        y_test.append(label)
# 데이터를 학습시키고 예측하기 --- (※4)
    clf = svm.SVC()
    
    # fit(학습용_데이터, 레이블_데이터) 함수 : 데이터를 학습시킨다.
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)

# 정답률 구하기 --- (※5)
# accuracy_score 함수를 이용하면 정답률을 쉽게 구해준다.
# accuracy_score(정답, 예측_결과_배열)
ac_score = metrics.accuracy_score(y_test, prediction)

print("정답률 =", ac_score)
cl_report = metrics.classification_report( y_test, prediction)
print("\n리포트 =", cl_report)
