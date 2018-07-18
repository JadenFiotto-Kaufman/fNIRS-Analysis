import sys

import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QHBoxLayout, QLabel, QAction, QMenuBar, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle

from src.main.fNIRLib import fNIRLib


class Window(QDialog):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the plot or (figure)
        self.canvas = FigureCanvas(self.figure)

        self.openFileDialog()

        # this is the Navigation Toolbar for the top of the plot
        self.toolbar = NavigationToolbar(self.canvas, self)

        # create the layout and menu
        layout = QVBoxLayout()

        menu_bar = QMenuBar()
        fileMenu = menu_bar.addMenu('File')

        importDataAction = QAction('Import Data', self)
        importDataAction.setShortcut('Ctrl+I')
        importDataAction.triggered.connect(self.openFileDialog)

        fileMenu.addAction(importDataAction)

        layout.setMenuBar(menu_bar)
        layout.addWidget(self.toolbar)

        layout.addWidget(self.canvas)

        outputHBox = QHBoxLayout()

        # create the labels that we will need to access again
        self.brainStateLabel = QLabel("Brain State : ")
        self.brainStateLabel.setFixedSize(200, 20)
        self.accuracyLabel = QLabel("Accuracy of Prediction: ")
        self.accuracyLabel.setFixedSize(400, 20)

        predictionStatisticsVBoxLeft = QVBoxLayout()
        predictionStatisticsVBoxLeft.addWidget(self.brainStateLabel)

        predictionStatisticsVBoxRight = QVBoxLayout()
        predictionStatisticsVBoxRight.addWidget(self.accuracyLabel)

        outputHBox.addLayout(predictionStatisticsVBoxLeft)
        outputHBox.addLayout(predictionStatisticsVBoxRight)

        layout.addLayout(outputHBox)
        self.setLayout(layout)

        # mouse button press event
        press_event_id = self.canvas.mpl_connect('button_press_event', self.onclick)
        # motion_event_id = self.canvas.mpl_connect('motion_notify_event', self.on_move)

        self.plot()

        self.model = fNIRLib.load_model()

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(self, caption="Import Processed Data Folder")
        if folder_path:
            print(folder_path)
            data = fNIRLib.importSingleton(folder_path + "/TRAIN_DATA")

            data_frames = []
            for df in data:
                data_frames.append(df)

            all_features = pd.concat(data_frames)
            self.features = all_features.drop(all_features.columns[[16]], axis=1)
            self.plot()

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.figure.tight_layout()

        if event.button == 3:
            self.draw_rectangle(event.xdata, event.ydata)

            self.brainStateLabel.setText("Brain State: High")
            self.accuracyLabel.setText("Prediction Accuracy: 75%")
            QApplication.processEvents()
            x_point = int(round(event.xdata))

            ml_data = self.features.iloc[x_point]

            print(self.model.predict_classes(ml_data))
        self.canvas.draw()


    # def on_move(self, event):
    #     # get the x and y pixel coords
    #     x, y = event.x, event.y
    #
    #     if event.inaxes:
    #         ax = event.inaxes  # the axes instance
    #         print('data coords %f %f' % (event.xdata, event.ydata))
    #     self.figure.tight_layout()

    def draw_rectangle(self, x, y):
        self.ax.patches.clear()
        rectangle = Rectangle((0, -10), width=x, height=100, color='#0F0F0F2F')
        self.ax.add_patch(rectangle)
        self.canvas.draw()

    def plot(self):
        self.figure.clear()

        # create an axis
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('auto')

        self.ax.plot(self.features, '-')
        self.ax.set_xlim(0, len(self.features))
        self.figure.tight_layout()

        # refresh canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
