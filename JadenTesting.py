import numpy
from keras.layers import LSTM, Conv1D, Conv2D
from keras.layers import Dense, Flatten, TimeDistributed, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from src.main.fNIRLib import fNIRLib

data = fNIRLib.importData("processed/",combine=True,points=True)
features, classes = fNIRLib.xySplit(data.iloc[0])
scaled = fNIRLib.minMaxScale(features)

xTrain, xTest, yTrain, yTest = fNIRLib.testTrain(features,classes, size=.222)
yTrain = yTrain.values
yTest = yTest.values
xTrain = fNIRLib.to3D(xTrain)
xTest = fNIRLib.to3D(xTest)
input_shape = xTrain.shape

model = Sequential()
model.add(Conv1D(filters=60, kernel_size=10, padding='same', activation='relu',input_shape=input_shape[1:]))
model.add(MaxPooling1D(pool_size=5))
model.add(LSTM(100, dropout=0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Save the checkpoint in the /output folder
filepath = 'my_model_weights.hdf5'

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_acc',
                            verbose=0,
                            save_best_only=True,
                            mode='max')
#model.load_weights(filepath)
model.fit(xTrain, yTrain, epochs=10000, batch_size=80,validation_data=(xTest, yTest),callbacks=[checkpoint])