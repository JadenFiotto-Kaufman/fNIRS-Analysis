from src.main.fNIR import fNIR
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()
if __name__ == "__main__":
    fNIR.train("../../processed/", "DNN", epochs=2000, batch_size=3, combine=True, test_size=.1, load=True, scale=True)