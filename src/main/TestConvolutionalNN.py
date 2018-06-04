import numpy
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential, model_from_json

from src.main.fNIRLib import fNIRLib

p13TestData = fNIRLib.importData("/Users/corpa/BCI/processed/").get("p13")

xTest = p13TestData.iloc[:, 0:16]
yTest = p13TestData.iloc[:, 16]

xTest = numpy.expand_dims(xTest, axis=2)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

score = loaded_model.evaluate(xTest, yTest, batch_size=16, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))