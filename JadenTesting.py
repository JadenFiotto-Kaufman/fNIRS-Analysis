
from tsfresh import extract_relevant_features
import pandas as pd

# def tsextract(data, classes):
#     tsdata = []
#     for i, d in enumerate(data):
#         length = d.shape[0]
#         d['id'] = [i] * length
#         d['time'] = range(length)
#         tsdata.append(d)
#     ts_dataframe = pd.concat(tsdata)
#     extracted_features = extract_relevant_features(ts_dataframe, classes, column_id="id", column_sort="time")
#     extracted_features.to_pickle('tsfeatures.pkl')

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
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
    xTrain, xTest, yTrain, yTest = fNIRLib.testTrain(data,classes, size=.1, seed=seed)
    # yTrain = yTrain.values
    # yTest = yTest.values
    # xTrain = xTrain.values
    # xTest = xTest.values
    # xTrain = fNIRLib.to3D(xTrain)
    # xTest = fNIRLib.to3D(xTest)
    # input_shape = xTrain.shape
    model = Sequential()
    model.add(Dense(20, input_dim=229, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = 'my_model_weights.hdf5'


    checkpoint = ModelCheckpoint(filepath,
                                monitor='val_acc',
                                verbose=0,
                                save_best_only=True,
                                mode='max')
    #model.load_weights(filepath)
    model.fit(xTrain, yTrain, epochs=5000, batch_size=3,validation_data=[xTest, yTest],callbacks=[checkpoint], shuffle=True)
    # preds = model.predict_classes(xTrain)
    # print(preds)
    # print(classes)