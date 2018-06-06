import numpy
from keras.layers import LSTM, Conv1D, Conv2D
from keras.layers import Dense, Flatten, TimeDistributed
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
model.add(LSTM(50, activation='sigmoid', return_sequences=True, dropout=0.5, batch_input_shape=(68, input_shape[1], input_shape[2])))
model.add(LSTM(50, activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=5000, batch_size=68)

