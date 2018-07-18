

import pandas as pd
from json import dump
# def tsextract(data, classes):


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm, ensemble
    from keras.layers import LSTM, Conv1D
    from keras.layers import Dense, Flatten, TimeDistributed, MaxPooling1D, Dropout
    from keras.models import Sequential
    from keras.callbacks import ModelCheckpoint
    from src.main.fNIRLib import fNIRLib
    seed=10
    data = fNIRLib.importData("processed/",combine=True,points=True)
    features, classes = fNIRLib.xySplit(data.iloc[0])
    # features = fNIRLib.minMaxScale(features)
    #
    data = pd.read_pickle('tsfeatures.pkl')
    # sc = StandardScaler()
    # data = sc.fit_transform(data)
    xTrain, xTest, yTrain, yTest = fNIRLib.testTrain(data,classes, size=.2, seed=seed)

    featureNames = feature_extraction.settings.from_columns(data)
    with open("relevant_features.json", "w") as jsonFile:
        dump(featureNames, jsonFile)

    ayy = feature_extraction.extract_features(xTest, default_fc_parameters=featureNames)
    print(ayy)

    # clf = svm.SVC()
    # clf.fit(xTrain.values, yTrain.values)
    # z = clf.predict(xTest.values)
    # print(sum(z == yTest)/len(yTest))
    #
    # clf = ensemble.RandomForestClassifier(max_depth=2, random_state=seed)
    # clf.fit(xTrain.values, yTrain.values)
    # z = clf.predict(xTest.values)
    # print(sum(z == yTest) / len(yTest))

    # yTrain = yTrain.values
    # yTest = yTest.values
    # xTrain = xTrain.values
    # xTest = xTest.values
    # xTrain = fNIRLib.to3D(xTrain)
    # xTest = fNIRLib.to3D(xTest)
    # input_shape = xTrain.shape
    model = Sequential()
    model.add(Dense(25, input_dim=229, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    modeljson = model.to_json()
    with open("model_num.json", "w") as json_file:
        json_file.write(modeljson)
    filepath = 'my_model_weights.hdf5'


    checkpoint = ModelCheckpoint(filepath,
                                monitor='val_acc',
                                verbose=0,
                                save_best_only=True,
                                mode='max')
    model.load_weights(filepath)
    model.fit(xTrain, yTrain, epochs=5000, batch_size=3,validation_data=[xTest, yTest],callbacks=[checkpoint], shuffle=True)
    # preds = model.predict_classes(xTrain)
    # print(preds)
    # print(classes)