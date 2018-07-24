from src.main.fNIR import fNIR
import os
if __name__ == "__main__":
    #fNIR.train("../../processed/", "DNN", epochs=2000, batch_size=3, combine=True, test_size=.1, load=True, scale=True)
    fNIR.predict()
    #fNIR.predother()