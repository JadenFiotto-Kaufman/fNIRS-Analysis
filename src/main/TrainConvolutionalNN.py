import numpy
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, Conv3D, MaxPooling3D
from keras.layers import Dense, Flatten
from keras.models import Sequential

from src.main.fNIRLib import fNIRLib

data = fNIRLib.importData("../../processed/", combine=True, points=True)
features, classes = fNIRLib.xySplit(data.iloc[0])
scaled = fNIRLib.minMaxScale(features)

xTrain, xTest, yTrain, yTest = fNIRLib.testTrain(features,classes)

yTrain = yTrain.values
yTest = yTest.values
xTrain = fNIRLib.to3D(xTrain)
xTest = fNIRLib.to3D(xTest)

model = Sequential()

model.add(Conv2D(64, 2, activation='relu', input_shape=(input_shape[1], input_shape[2])))
# model.add(Conv2D(64, 2, activation='relu'))
model.add(MaxPooling2D())
# model.add(Conv2D(128, 2, activation='relu'))
model.add(Conv2D(128, 2, activation='relu'))
# model.add(GlobalAveragePooling2D())
Flatten()
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(xTrain, yTrain, batch_size=16, epochs=1000)

scores = model.evaluate(xTrain, yTrain, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))