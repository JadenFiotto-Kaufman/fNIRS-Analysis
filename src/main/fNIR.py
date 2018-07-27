from sklearn import svm, ensemble, preprocessing
from sklearn.externals import joblib
from keras.layers import Dense, MaxPooling1D, Dropout, LSTM, Conv1D
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
from src.main.fNIRLib import fNIRLib
import pandas as pd
from tsfresh import extract_relevant_features, feature_extraction
from json import dump, load
import os

class fNIR:
    seed = 11
    @staticmethod
    def tstag(features):
        '''
        Method to prepare timeseries data for tsfresh feature extraction
        :param features: Type of |Series(Dataframe(2D))|
        :return: Dataframe(2D)
        '''
        ts_features = []
        #Iterate through each timeseries
        for i, task in enumerate(features):
            length = task.shape[0]
            #Unique id for each task in same order as classes i.e task with id==0 has class at index 0 in classes
            task['id'] = [i] * length
            #Column for the time step, for this data as is shoudl be (0-259)
            task['time'] = range(length)
            ts_features.append(task)
        #Combine all together
        return pd.concat(ts_features)
    @staticmethod
    def genFeatures(features, classes):
        '''
        Method to generate features, particularlly for entire dataset as it take a long time so as to save it in tsfeatures.pkl
        :param features: Type of |Series(Dataframe(2D))|
        :param classes: Type of |Series(integers)|
        :return: Type of |Dataframe(2D)| columns are the extracted features, rows designate each timeseries
        '''
        #Prepares the data for feature extraction
        ts_dataframe = fNIR.tstag(features)
        #Ts fresh function to extract all features and based on classes for classificatiom, choose the mose relevant ones
        extracted_features = extract_relevant_features(ts_dataframe, classes, column_id="id", column_sort="time")
        #Saves the feautes to pickle files so we dont have to generate it again
        extracted_features.to_pickle('tsfeatures.pkl')
        #Saves the names of the features for future extraction for prediction
        featureNames = feature_extraction.settings.from_columns(extracted_features)
        with open("features_extracted.json", "w") as jsonFile:
            dump(featureNames, jsonFile)
        return extracted_features
    @staticmethod
    def preprocess(filepath, combine=True, extract=True, test_size=.2, scale=True):
        '''
        Completely preproceeses the data for machine learning depending on the type of problem
        :param filepath: Type of |String|
        :param combine: Boolean for combining all tasks indescriminately
        :param extract: Boolean fro extraction of features, only False if using LSTM
        :param test_size: Size of the validation data opposite the training data
        :param scale: Boolean for using the standard scaler
        :return: Type of |Dictionary('xTest','xTrain','yTest','yTrain')|
        '''
        data = fNIRLib.importData(filepath, combine=combine)
        features, classes = fNIRLib.xySplit(data)
        if extract:
            if combine:
                try:
                    features = pd.read_pickle('tsfeatures.pkl')
                except:
                    features = fNIR.genFeatures(features, classes)
            else:
                featurenames = None
                with open("features_extracted.json") as data_file:
                    featurenames = load(data_file)
                features = fNIR.tstag(features)
                features = feature_extraction.extract_features(features, kind_to_fc_parameters=featurenames, column_id="id", column_sort="time")
            #Order feature columns by name to standardie it for prediction
            features = features.reindex(sorted(list(features)), axis=1)
        if scale:
            scaler = preprocessing.StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features), columns=list(features))
            joblib.dump(scaler, "fNIRscaler.pkl")

        xTest, xTrain, yTest, yTrain = fNIRLib.testTrain(features, classes, size=test_size, seed=fNIR.seed)
        return {'xTest' : xTest,
                'xTrain' : xTrain,
                'yTrain' : yTrain,
                'yTest' : yTest}
    @staticmethod
    def neuralNet(model,data, epochs, batch_size, load,filepath = 'my_model_weights.hdf5'):
        '''
        Runs the neural network with the given data and saves the weights each epoch
        :param model: Keras model
        :param data: Dictionary of data returned from preprocessing
        :param epochs: Number of epochs
        :param batch_size: Number of tasks per batch
        :param load: Boolean to load model weighs from file
        :param filepath: String of file path
        :return: None
        '''
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
        '''
        Main function to train the network 
        :param filepath: Sting of filepath to data folder
        :param method: String of which method to use, DNN, LSTM, RDF, SVM
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param combine: Boolean to combine tasks
        :param test_size: Size of validation data
        :param load: Boolean to load model or start a new one
        :param scale: Boolean to scale data
        :return: None
        '''
        model = Sequential()
        if method == "DNN":
            '''
            The main method of machine learning to extract the features and run it through various DNNs
            '''
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
            '''
            Other neural network method of using the data as is
            '''
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
            '''
            Basic support vector machine method for comparision
            '''
            data = fNIR.preprocess(filepath, combine=True, extract=True, test_size=test_size, scale=scale)
            clf = svm.SVC()
            clf.fit(data['xTrain'].values, data['yTrain'].values)
            z = clf.predict(data['xTest'].values)
            print(sum(z == data['yTest'])/len(data['yTest']))
        elif method == "RDF":
            '''
            Basic random decision forest method for comparision
            '''
            data = fNIR.preprocess(filepath, combine=True, extract=True, test_size=test_size, scale=scale)
            clf = ensemble.RandomForestClassifier(max_depth=2, random_state=fNIR.seed)
            clf.fit(data['xTrain'].values, data['yTrain'].values)
            z = clf.predict(data['xTest'].values)
            print(sum(z == data['yTest']) / len(data['yTest']))
            pass
    @staticmethod
    def predict(filepath, weight_path='my_model_weights.hdf5'):
        '''
        Method to predict just one task meaning it will take just the last 260 time steps
        :param filepath: String of filepath to data file
        :param weight_path: Path to model weights
        :return: None
        '''
        featurenames = None
        with open("features_extracted.json") as data_file:
            featurenames = load(data_file)

        scaler = joblib.load("fNIRscaler.pkl")

        #print(features)
        model = Sequential()
        model.add(Dense(25, input_dim=229, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(1, activation='sigmoid'))

        model.load_weights(weight_path)

        data = pd.read_csv(filepath, usecols=range(4, 21), names=fNIRLib.column_names)
        data = data.tail(260)
        features, classes = (data.iloc[:, :-1], data.iloc[0, -1])
        features['id'] = [999] * 260
        features['time'] = range(260)
        features = feature_extraction.extract_features(features, kind_to_fc_parameters=featurenames, column_id="id",column_sort="time")
        features = pd.DataFrame(scaler.transform(features), columns=list(features))
        prediction = model.predict_classes(features)

        print(prediction[0][0])

    @staticmethod
    def load_model():
        json_file = open('../../results/models/model_num.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("../../results/models/25_20.hdf5")

        return loaded_model

