import numpy
from keras.layers import LSTM, Conv1D, Conv2D, Dropout
from keras.layers import Dense, Flatten, TimeDistributed, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from src.main.fNIRLib import fNIRLib

models = []
data = fNIRLib.importData("../../processed/", combine=True, points=True)
features, classes = fNIRLib.xySplit(data.iloc[0])
scaled = fNIRLib.minMaxScale(features)

xTrain, xTest, yTrain, yTest = fNIRLib.testTrain(scaled, classes, size=.222)
yTrain = yTrain.values
yTest = yTest.values
xTrain = fNIRLib.to3D(xTrain)
xTest = fNIRLib.to3D(xTest)
input_shape = xTrain.shape


def trainModel(i, startIndex):
    model = Sequential()
    model.add(Conv1D(filters=60, kernel_size=10, padding='same', activation='relu', input_shape=input_shape[1:]))
    model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(100, dropout=0.5))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Save the checkpoint in the /output folder
    filepath = '../../results/models/weights' + str(i) + '.hdf5'

    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='max')


    # model.load_weights(filepath)
    model.fit(xTrain[startIndex:startIndex + 20], yTrain[startIndex:startIndex + 20],
              epochs=500, batch_size=80,
              validation_data=(xTest[startIndex:startIndex + 20], yTest[startIndex:startIndex + 20]),
              callbacks=[checkpoint])
    return model


for i in range(1, 3):
    trainedModel = trainModel(i, i * 20)
    models.append(trainedModel)

for i in range(1, 3):
    model = models[i]
    index = i * 20
    scores = model.evaluate(xTrain[index: index + 20],
                            yTrain[index: index + 20],
                            verbose=0)

    print("%s[%d]: %.2f%%" % (model.metrics_names[1], i, scores[1] * 100))
