from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
import json

max_words = 67395 # 입력 단어 수: word-dic.json 파일 참고
max_words = 56681 # 입력 단어 수: word-dic.json 파일 참고
nb_classes = 9    # 9개의 카테고리
batch_size = 64 
nb_epoch = 20

# MLP 모델 생성하기 --- (※1)
def build_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model

# 데이터 읽어 들이기--- (※2)
data = json.load(open("./newstext/data-mini.json")) 
# data = json.load(open("./newstext/data.json"))

X = data["X"] # 텍스트를 나타내는 데이터
Y = data["Y"] # 카테고리 데이터

# 학습하기 --- (※3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
Y_train = np_utils.to_categorical(Y_train, nb_classes)

print(len(X_train),len(Y_train))

model = KerasClassifier(
    build_fn=build_model, 
    nb_epoch=nb_epoch, 
    batch_size=batch_size)

model.fit(X_train, Y_train)
# 예측하기 --- (※4)
y = model.predict(X_test)
ac_score = metrics.accuracy_score(Y_test, y)
cl_report = metrics.classification_report(Y_test, y)

print("정답률 =", ac_score)
print("리포트 =\n", cl_report)

'''
data-mini.json으로 테스트시 결과

Using TensorFlow backend.
99 99
Epoch 1/1
2018-01-11 00:16:44.013600: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2

64/99 [==================>...........] - ETA: 0s - loss: 2.2389 - acc: 0.1562
99/99 [==============================] - 2s 20ms/step - loss: 2.0614 - acc: 0.2525
정답률 = 0.575757575758
리포트 =
              precision    recall  f1-score   support

          0       0.88      1.00      0.93         7
          1       1.00      0.14      0.25         7
          2       1.00      0.12      0.22         8
          3       0.29      1.00      0.44         2
          4       0.43      1.00      0.60         3
          5       0.56      0.83      0.67         6

avg / total       0.80      0.58      0.51        33

'''