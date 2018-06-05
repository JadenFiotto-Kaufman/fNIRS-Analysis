import numpy
from keras.layers import LSTM, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential

from src.main.fNIRLib import fNIRLib

data = fNIRLib.importData("processed/",combine=True,points=True)
features, classes = fNIRLib.xySplit(data.iloc[0])
scaled = fNIRLib.minMaxScale(features)

xTrain, xTest, yTrain, yTest = fNIRLib.testTrain(features,classes)
yTrain = yTrain.values
yTest = yTest.values
xTrain = fNIRLib.to3D(xTrain)
xTest = fNIRLib.to3D(xTest)
input_shape = xTrain.shape

model = Sequential()
model.add(LSTM(50, input_shape=(input_shape[1], input_shape[2])))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=3000, batch_size=64)

