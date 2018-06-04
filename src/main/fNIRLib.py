import tensorflow as tf
import pandas as pd
import os

class fNIRLib:
    @staticmethod
    def importData(data_path):
        TEST_subjects = {}
        names = ["a1HbO", "a1Hb", "a2HbO", "a2Hb", "a3HbO", "a3Hb", "a4HbO", "a4Hb", "b1HbO", "b1Hb", "b2HbO", "b2Hb", "b3HbO", "b3Hb", "b4HbO", "b4Hb", "Class"]
        for subject in os.listdir(data_path):
            if subject != ".DS_Store":
                TEST_subjects[subject] = pd.read_csv(data_path + subject + "/TRAIN_DATA", usecols=range(4,21), names=names)
        return TEST_subjects