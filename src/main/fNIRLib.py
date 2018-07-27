from sklearn import preprocessing
import pandas as pd
from numpy import random
from numpy import array
import os

class fNIRLib:
    #Number of timesteps per task
    tslength = 260
    #Names of each channel plus a column for the type of cognitive load
    column_names = ["a1HbO", "a1Hb", "a2HbO", "a2Hb", "a3HbO", "a3Hb", "a4HbO", "a4Hb", "b1HbO", "b1Hb", "b2HbO", "b2Hb",
                 "b3HbO", "b3Hb", "b4HbO", "b4Hb", "Class"]
    @staticmethod
    def importSingleton(filename):
        data = pd.read_csv(filename, usecols=range(4, 21), names=fNIRLib.column_names)
        data = pd.Series([data.iloc[fNIRLib.tslength * x:fNIRLib.tslength * (x + 1), :] for x in range(data.shape[0] // fNIRLib.tslength)])
        return data
    @staticmethod
    def importData(data_path, combine=False):
        '''
        Function to import each subject data
        :param data_path: relative path to /processed/ folder
        :param combine: option to disregard the fact that tasks belong to separate subjects and combine all of them
        :return: Type of |list(DataFrame(2D)))| shape: (Tasks, (Time Steps, Features))
        '''
        train = [pd.read_csv(data_path + subject + "/TRAIN_DATA", usecols=range(4,21), names=fNIRLib.column_names) for subject in os.listdir(data_path) if subject != ".DS_Store"]
        test = [pd.read_csv(data_path + subject + "/TEST_DATA", usecols=range(4,21), names=fNIRLib.column_names) for subject in os.listdir(data_path) if subject != ".DS_Store"]
        data = train + test
        data = [[d.iloc[fNIRLib.tslength * x:fNIRLib.tslength * (x + 1),:] for x in range(d.shape[0] // fNIRLib.tslength)] for d in data]
        if combine:
            data = [x for y in data for x in y]
        else:
            data = data[0]
        return data
    @staticmethod
    def xySplit(data):
        '''
        Splits data into features and classes
        :param data: Type of |list(DataFrames(2D))| shape: (Tasks, (Time Steps, Features))
        :return: Type of |(Series(DataFrame(2D))), Series(Integers)| shape: (Tasks, (Time Steps, Features)), (Classes for each task,)
        '''
        split_data = [(y.iloc[:,:-1], y.iloc[0,-1]) for y in data]
        x_data = pd.Series([y[0] for y in split_data])
        y_data = pd.Series([y[1] for y in split_data])
        return x_data, y_data
    @staticmethod
    def Scale(data, scaler=None):
        '''
        Scales data based on all readings per column bases on given scaler, or StandardScaler if None is given
        :param data: Type of |Series(DataFrames(2D))| shape: (Tasks, (Time Steps, Features))
        :param scaler: Type of | sklearn.preprocessing.(Some kind of scaler)|
        :return: Type of |(Series(DataFrame(2D)))| shape: (Tasks, (Time Steps, Features))
        '''
        names = list(data.iloc[0])
        if scaler is None:
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(pd.concat([d for d in data]))
        return pd.Series([pd.DataFrame(scaler.transform(y), columns=names) for y in data])
    @staticmethod
    def to3D(data):
        '''
        Converts data from pandas dataset format to numpy array format for full timeseries
        :param data: Type of |Series(DataFrames(2D))| shape: (Reading/Task, (Time Steps, Features))
        :return: Type of |Numpy.array(3D)| shape: (Subjects, Reading/Task, (Time Steps, Features))
        '''
        return array([x.values for x in data])
    @staticmethod
    def testTrain(features, classes, size=1./3., seed=10):
        '''
        Divides data into test and train based on percentage reserved for testing
        :param features: Type of |Series(DataFrames(2D))| shape: (Tasks, (Time Steps, Features))
        :param classes: Type of |Series(integers)| shape: (Classes,)
        :param size: Type of |double between 0-1| Percent of data reserved for testing
        :param seed: Type of |integer| Seed for random index selection
        :return: Train features, test features, train classes, test classes
        '''
        random.seed(seed)
        n = features.shape[0]
        index = random.choice(range(n),replace=False, size=int(n * size))
        return features.iloc[index].values, features.drop(index).values, classes.iloc[index], classes.drop(index)




