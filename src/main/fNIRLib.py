from sklearn import preprocessing
import pandas as pd
from numpy import random, arange, array
import os
from math import floor


class fNIRLib:
    @staticmethod
    def importData(data_path, combine=False, points=False):
        subjects = []
        names = ["a1HbO", "a1Hb", "a2HbO", "a2Hb", "a3HbO", "a3Hb", "a4HbO", "a4Hb", "b1HbO", "b1Hb", "b2HbO", "b2Hb",
                 "b3HbO", "b3Hb", "b4HbO", "b4Hb", "Class"]
        for subject in os.listdir(data_path):
            if subject != ".DS_Store":
                subjects.append(pd.read_csv(data_path + subject + "/TRAIN_DATA", usecols=range(4, 21), names=names))
        data = pd.Series(subjects)
        if combine:
            data = pd.Series([pd.concat(data.values)])
        data = pd.Series([pd.Series([s]) for s in data])
        if points:
            data = pd.Series(
                [pd.Series([d.iloc[0].iloc[260 * x:260 * (x + 1), :] for x in range(d.iloc[0].shape[0] // 260)]) for d
                 in data])
        return data

    @staticmethod
    def xySplit(data):
        split_data = [(y.iloc[:, :-1], y.iloc[0, -1]) for y in data]
        x_data = pd.Series([y[0] for y in split_data])
        y_data = pd.Series([y[1] for y in split_data])
        return x_data, y_data

    @staticmethod
    def minMaxScale(data):
        names = list(data.iloc[0])
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        return pd.Series([pd.DataFrame(scaler.fit_transform(y.values), columns=names) for y in data])

    @staticmethod
    def to3D(data):
        return array([x.values for x in data])

    @staticmethod
    def testTrain(features, classes, size=1. / 3.):
        test = features.sample(frac=size, random_state=200)
        return features.drop(test.index), test, classes.drop(test.index), classes.iloc[test.index]
