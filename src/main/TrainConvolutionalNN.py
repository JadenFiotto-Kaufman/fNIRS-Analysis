import numpy
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential

from src.main.fNIRLib import fNIRLib

p12TrainData = fNIRLib.importData("/Users/corpa/BCI/processed/").get("p12")

xTrain = p12TrainData.iloc[:, 0:16]
yTrain = p12TrainData.iloc[:, 16]

p13TestData = fNIRLib.importData("/Users/corpa/BCI/processed/").get("p13")

xTest = p13TestData.iloc[:, 0:16]
yTest = p13TestData.iloc[:, 16]


model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(16, 1)))
model.add(Conv1D(64, 2, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(128, 2, activation='relu'))
model.add(Conv1D(128, 2, activation='relu'))
model.add(GlobalAveragePooling1D())
Flatten()
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# was having trouble with input shape, now it should be correctly looking at the shape.
xTrain = numpy.expand_dims(xTrain, axis=2)

model.fit(xTrain, yTrain, batch_size=16, epochs=100)

scores = model.evaluate(xTrain, yTrain, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

xTest = numpy.expand_dims(xTest, axis=2)

score = model.evaluate(xTest, yTest, batch_size=16)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))