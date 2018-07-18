from sklearn.preprocessing import StandardScaler
from sklearn import svm, ensemble
from keras.layers import LSTM, Conv1D
from keras.layers import Dense, Flatten, TimeDistributed, MaxPooling1D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from src.main.fNIRLib import fNIRLib
import pandas as pd
from tsfresh import extract_relevant_features, feature_extraction
from json import dump

class fNIR:
    @staticmethod
    def genFeatures(features, classes):
        tsdata = []
        for i, d in enumerate(features):
            length = d.shape[0]
            d['id'] = [i] * length
            d['time'] = range(length)
            tsdata.append(d)
        ts_dataframe = pd.concat(tsdata)
        extracted_features = extract_relevant_features(ts_dataframe, classes, column_id="id", column_sort="time")
        extracted_features.to_pickle('tsfeatures.pkl')
        featureNames = feature_extraction.settings.from_columns(extracted_features)
        with open("relevant_features.json", "w") as jsonFile:
            dump(featureNames, jsonFile)
        return True
    @staticmethod
    def preprocess(filepath, combine=True, points=True, extract=False):
        data = fNIRLib.importData(filepath, combine=combine)
        features, classes = fNIRLib.xySplit(data)
        if extract:
            try:
                features = pd.read_pickle('tsfeatures.pkl')
            except:
                fNIR.genFeatures(features, classes)
                features = pd.read_pickle('tsfeatures.pkl')

    @staticmethod
    def neuralNet(model,data):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    @staticmethod
    def train(filepath, method="DNN", epochs=5000, batch_size=3, combine=True):
        model = Sequential()
        if method == "DNN":

            model.add(Dense(25, input_dim=data.shape[1], activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(20, activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(1, activation='sigmoid'))
            fNIR.neuralNet(model, data)
        elif method == "LSTM":
        elif method == "SVM":
        elif method == "RDF":
