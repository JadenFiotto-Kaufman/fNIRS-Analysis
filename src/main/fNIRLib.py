from sklearn import preprocessing
import pandas as pd
from numpy import random
from numpy import array
import os

class fNIRLib:
    tslength = 260
    @staticmethod
    def importSingleton(filename):
        names = ["a1HbO", "a1Hb", "a2HbO", "a2Hb", "a3HbO", "a3Hb", "a4HbO", "a4Hb", "b1HbO", "b1Hb", "b2HbO", "b2Hb",
                 "b3HbO", "b3Hb", "b4HbO", "b4Hb", "Class"]
        data = pd.read_csv(filename, usecols=range(4, 21), names=names)
        data = pd.Series([data.iloc[260 * x:260 * (x + 1), :] for x in range(data.shape[0] // 260)])
        return data
    @staticmethod
    def importData(data_path, combine=False):
        '''
        Function to import each subject data
        :param data_path: relative path to /processed/ folder
        :param combine: option to disregard differentiation of subjects and combine into one dataframe (recommended)
        :param points: option to group each task into it's own data frame. This is every 260 time steps (recommended)
        :return: Type of |Series(Series(DataFrame(2D)))| shape: (Subjects, Reading/Task, (Time Steps, Features))
        '''
        names = ["a1HbO", "a1Hb", "a2HbO", "a2Hb", "a3HbO", "a3Hb", "a4HbO", "a4Hb", "b1HbO", "b1Hb", "b2HbO", "b2Hb", "b3HbO", "b3Hb", "b4HbO", "b4Hb", "Class"]
        train = [pd.read_csv(data_path + subject + "/TRAIN_DATA", usecols=range(4,21), names=names) for subject in os.listdir(data_path) if subject != ".DS_Store"]
        test = [pd.read_csv(data_path + subject + "/TEST_DATA", usecols=range(4,21), names=names) for subject in os.listdir(data_path) if subject != ".DS_Store"]
        data = train + test
        data = [[d.iloc[260 * x:260 * (x + 1),:] for x in range(d.shape[0] // 260)] for d in data]
        if combine:
            data = [x for y in data for x in y]
        else:
            data = data[0]
        return data
    @staticmethod
    def xySplit(data):
        '''
        Splits data into features and classes
        :param data: Type of |Series(DataFrames(2D))| shape: (Reading/Task, (Time Steps, Features))
        :return: Type of |Series(Series(DataFrame(2D)))| shape: (Subjects, Reading/Task, (Time Steps, Features))
        '''
        split_data = [(y.iloc[:,:-1], y.iloc[0,-1]) for y in data]
        x_data = pd.Series([y[0] for y in split_data])
        y_data = pd.Series([y[1] for y in split_data])
        return x_data, y_data
    @staticmethod
    def minMaxScale(data):
        '''
        Scales data based on all readings
        :param data: Type of |Series(DataFrames(2D))| shape: (Reading/Task, (Time Steps, Features))
        :return: Type of |Series(Series(DataFrame(2D)))| shape: (Subjects, Reading/Task, (Time Steps, Features))
        '''
        names = list(data.iloc[0])
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(pd.concat([d for d in data]))
        return pd.Series([pd.DataFrame(scaler.transform(y), columns=names) for y in data])
    @staticmethod
    def to3D(data):
        '''
        Converts data from pandas dataset format to numpy array format
        :param data: Type of |Series(DataFrames(2D))| shape: (Reading/Task, (Time Steps, Features))
        :return: Type of |Numpy.array(3D)| shape: (Subjects, Reading/Task, (Time Steps, Features))
        '''
        return array([x.values for x in data])
    @staticmethod
    def testTrain(features, classes, size=1./3., seed=10):
        '''
        Divides data into test and train based on percentage reserved for testing
        :param features: Type of |Series(DataFrames(2D))| shape: (Reading/Task, (Time Steps, Features))
        :param classes: Type of |Series()| shape: (Classes)
        :param size: Percent of data reserved for testing
        :return: Train features, test features, train classes, test classes
        '''
        random.seed(seed)
        n = features.shape[0]
        index = random.choice(range(n),replace=False, size=int(n * size))
        return features.iloc[index], features.drop(index), classes.iloc[index], classes.drop(index)




