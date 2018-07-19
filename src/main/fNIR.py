from sklearn import svm, ensemble, preprocessing
from keras.layers import Dense, MaxPooling1D, Dropout, LSTM, Conv1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from src.main.fNIRLib import fNIRLib
import pandas as pd
from tsfresh import extract_relevant_features, feature_extraction
from json import dump, load

class fNIR:
    seed = 11
    @staticmethod
    def tstag(features):
        tsdata = []
        for i, d in enumerate(features):
            length = d.shape[0]
            d['id'] = [i] * length
            d['time'] = range(length)
            tsdata.append(d)
        return pd.concat(tsdata)
    @staticmethod
    def genFeatures(features, classes):
        ts_dataframe = fNIR.tstag(features)
        extracted_features = extract_relevant_features(ts_dataframe, classes, column_id="id", column_sort="time")
        extracted_features.to_pickle('tsfeatures.pkl')
        featureNames = feature_extraction.settings.from_columns(extracted_features)
        with open("features_extracted.json", "w") as jsonFile:
            dump(featureNames, jsonFile)
        return True
    @staticmethod
    def preprocess(filepath, combine=True, extract=False, test_size=.2, scale = True):
        data = fNIRLib.importData(filepath, combine=combine)
        features, classes = fNIRLib.xySplit(data)
        if extract:
            if combine:
                try:
                    features = pd.read_pickle('tsfeatures.pkl')
                except:
                    fNIR.genFeatures(features, classes)
                    features = pd.read_pickle('tsfeatures.pkl')
            else:
                featurenames = None
                with open("features_extracted.json") as data_file:
                    featurenames = load(data_file)
                features = fNIR.tstag(features)
                features = feature_extraction.extract_features(features, kind_to_fc_parameters=featurenames, column_id="id", column_sort="time")
        if scale:
            scaler = preprocessing.StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features), columns=list(features))
        xTest, xTrain, yTest, yTrain = fNIRLib.testTrain(features, classes, size=test_size, seed=fNIR.seed)
        return {'xTest' : xTest,
                'xTrain' : xTrain,
                'yTrain' : yTrain,
                'yTest' : yTest}
    @staticmethod
    def neuralNet(model,data, epochs, batch_size, load,filepath = 'my_model_weights.hdf5'):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='max')
        if load:
            model.load_weights(filepath)
        model.fit(data['xTrain'], data['yTrain'], epochs=epochs, batch_size=batch_size, validation_data=[data['xTest'], data['yTest']], callbacks=[checkpoint],shuffle=True)

    @staticmethod
    def train(filepath, method, epochs=5000, batch_size=3, combine=True, test_size=.2, load=False, scale=True):
        model = Sequential()
        if method == "DNN":
            data = fNIR.preprocess(filepath, combine=combine, extract=True, test_size=test_size, scale=scale)
            model.add(Dense(25, input_dim=data['xTrain'].shape[1], activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(20, activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(10, activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(10, activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(1, activation='sigmoid'))
            fNIR.neuralNet(model, data, epochs, batch_size, load)
        elif method == "LSTM":
            data = fNIR.preprocess(filepath, combine=combine, extract=False, test_size=test_size, scale=scale)
            data['xTrain'] = fNIRLib.to3D(data['xTrain'])
            data['xTest'] = fNIRLib.to3D(data['xTest'])
            model.add(Conv1D(50, kernel_size=3, activation='relu', input_shape=(data['xTrain'].shape[1:])))
            model.add(MaxPooling1D(5))
            model.add(Conv1D(32, kernel_size=3, activation='relu'))
            model.add(LSTM(50, dropout=.5, recurrent_dropout=.5))
            model.add(Dense(1, activation='sigmoid'))
            fNIR.neuralNet(model, data, epochs, batch_size, load)
        elif method == "SVM":
            data = fNIR.preprocess(filepath, combine=True, extract=True, test_size=.2)
            clf = svm.SVC()
            clf.fit(data['xTrain'].values, data['yTrain'].values)
            z = clf.predict(data['xTest'].values)
            print(sum(z == data['yTest'])/len(data['yTest']))
        elif method == "RDF":
            data = fNIR.preprocess(filepath, combine=True, extract=True, test_size=.2)
            clf = ensemble.RandomForestClassifier(max_depth=2, random_state=fNIR.seed)
            clf.fit(data['xTrain'].values, data['yTrain'].values)
            z = clf.predict(data['xTest'].values)
            print(sum(z == data['yTest']) / len(data['yTest']))
            pass
