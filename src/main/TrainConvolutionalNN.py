import numpy
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Dense, Flatten
from keras.models import Sequential

from src.main.fNIRLib import fNIRLib

trainData = fNIRLib.importData("/Users/corpa/BCI/processed/").get("p12")

xTrain = trainData.iloc[:, 0:16]
yTrain = trainData.iloc[:, 16]

print(xTrain.head())

print('y:')
print(yTrain.head())

model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(16, 1)))
model.add(MaxPooling1D())
model.add(Conv1D(64, 2, activation='relu'))
model.add(GlobalAveragePooling1D())
Flatten()
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# was having trouble with input shape, now it should be correctly looking at the shape.
xTrain = numpy.expand_dims(xTrain, axis=2)

model.fit(xTrain, yTrain, batch_size=16, epochs=100)
# score = model.evaluate(x_test, y_test, batch_size=16)