from keras.layers import Conv1D, MaxPooling1D, InputLayer, Dense, GlobalAveragePooling1D
from keras.models import Sequential

import numpy as np

from src.main.fNIRLib import fNIRLib

data = fNIRLib.importData("../../processed/", combine=True, points=True)
features, classes = fNIRLib.xySplit(data.iloc[0])
scaled = fNIRLib.minMaxScale(features)

xTrain, xTest, yTrain, yTest = fNIRLib.testTrain(features, classes)

yTrain = yTrain.values
yTest = yTest.values

xTrain = fNIRLib.to3D(xTrain)
xTest = fNIRLib.to3D(xTest)
input_shape = xTrain.shape

print(yTrain.shape)
print(xTrain.shape)

model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', batch_input_shape=(340, 16, 260)))
# model.add(Conv2D(64, 2, activation='relu'))
model.add(MaxPooling1D())
# model.add(Conv2D(128, 2, activation='relu'))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='linear'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

model.fit(xTrain, yTrain, batch_size=16, epochs=1000)

# scores = model.evaluate(xTrain, yTrain, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
