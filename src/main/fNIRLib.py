import tensorflow as tf
import pandas as pd
import os

class fNIRLib:
    @staticmethod
    def importData(data_path, combine=False, points=False):
        subjects = []
        names = ["a1HbO", "a1Hb", "a2HbO", "a2Hb", "a3HbO", "a3Hb", "a4HbO", "a4Hb", "b1HbO", "b1Hb", "b2HbO", "b2Hb", "b3HbO", "b3Hb", "b4HbO", "b4Hb", "Class"]
        for subject in os.listdir(data_path):
            if subject != ".DS_Store":
                subjects.append(pd.read_csv(data_path + subject + "/TRAIN_DATA", usecols=range(4,21), names=names))
        data = pd.Series(subjects)
        if combine:
            data = pd.Series([pd.concat(data.values).reset_index()])
        if points:
            data = pd.Series([pd.Series([d.iloc[260 * x:260 * (x + 1),:] for x in range(d.shape[0] // 260)]) for d in data])
        return data
    @staticmethod
    def xySplit(data):
        split_data = [[(y.iloc[:,:-1], y.iloc[0,-1]) for y in x] for x in data]
        x_data = pd.Series([pd.Series([y[0] for y in x]) for x in split_data])
        y_data = pd.Series([pd.Series([y[1] for y in x]) for x in split_data])
        return x_data, y_data
