'''
작성일 : 2020-09-11
작성자 : 이권동
코드 개요 : window, step 에 따른 학습 후 모델을 저장하는 코드
'''
from keras.utils import *
import tensorflow as tf
import numpy as np


window = 200
step = 20
path = "/home/dblab/pothole/data/npy/"

x = np.load(path + str(window) + "_" + str(step) + ".npz")
x_p = x["p"]
x_n = x["n"]

# 데이터 셔플 후 3 : 1 비율로 학습 데이터와 테스트 데이터를 나눔
# pothole 데이터가 압도적으로 적기 때문에 pothole 데이터 길이 기준으로
# 3 : 1 비율로 스플릿
shuffle = np.random.permutation(np.arange(len(x_n)))
shuffle2 = np.random.permutation(np.arange(len(x_p)))
x_p = x_p[shuffle2]
x_n = x_n[shuffle]
x_n = x_n[:len(x_p)]

d_p = len(x_p) // 4
d_n = len(x_n) // 4

X_train = np.concatenate([x_p[:d_p*3], x_n[:d_n*3]], axis=0)
X_valid = np.concatenate([x_p[d_p*3:], x_n[d_n*3:]], axis=0)

Y_train = np.concatenate([np.array([0 for _ in range(len(x_p[:d_p*3]))]), \
	np.array([1 for _ in range(len(x_n[:d_n*3]))])], axis=0)

Y_valid = np.concatenate([np.array([0 for _ in range(len(x_p[d_p*3:]))]), \
	np.array([1 for _ in range(len(x_n[d_n*3:]))])], axis=0)

X_train = X_train.reshape(-1, 1, window, 1)
X_valid = X_valid.reshape(-1, 1, window, 1)

Y_train = np_utils.to_categorical(Y_train, 2)
Y_valid = np_utils.to_categorical(Y_valid, 2)


# 모델 설정
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(1, 3), padding='same', activation='relu', input_shape=(1, window, 1) ))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(1, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=(1, 2), padding='same'))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(1, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(1, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=(1, 2), padding='same'))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(1, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(1, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=(1, 2), padding='same'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
#hist = model.fit( X_train, Y_train, batch_size=100, epochs=10, validation_data=(X_valid, Y_valid))
hist = model.fit( X_train, Y_train, batch_size=128, epochs=20) 

# acc
acc = model.evaluate(X_valid, Y_valid, batch_size = 128)
#print(hist)
model.summary()
print(hist.history["accuracy"][-1], acc[1])
#print(len(x_p[:d_p*3]), len(x_n[:d_n*3]), len(x_p[d_n*3:]), len(x_n[d_n*3:]))
# model save
#model.save("./model/" + str(window) + "_" + str(step) + ".h5")
#model.save("/home/dblab/pothole/data/model3/" + str(window) + "_" + str(step) + ".h5")
