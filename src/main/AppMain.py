import sys

import matplotlib.pyplot as plt
import numpy
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QHBoxLayout, QLabel, QAction, QMenuBar, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle, Circle, Ellipse

from src.main.fNIRLib import fNIRLib

class Window(QDialog):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.data = fNIRLib.importData("../../processed/", combine=True, points=True)
        self.features, self.classes = fNIRLib.xySplit(self.data.iloc[0])

        self.max_y = numpy.max(numpy.amax(self.features[0]))
        self.max_x = len(self.features[0])
        self.min_x = 0

        # self.factor = self.figure.get_figwidth() * (
        #         max(1.2 * self.features[0]) - min(1.2 * self.max_y)) / self.figure.get_figheight() / (self.max_x - self.min_x)

        self.setWindowTitle('fNIRS Analysis')

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QVBoxLayout()

        menu_bar = QMenuBar()
        fileMenu = menu_bar.addMenu('File')

        importDataAction = QAction('Import Data', self)
        importDataAction.setShortcut('Ctrl+I')
        importDataAction.triggered.connect(self.openFileDialog)

        fileMenu.addAction(importDataAction)

        layout.setMenuBar(menu_bar)
        layout.addWidget(self.toolbar)

        # canvasHBox = QHBoxLayout()
        # canvasHBox.addWidget(self.canvas)
        layout.addWidget(self.canvas)

        outputHBox = QHBoxLayout()
        brainStateLabel = QLabel("Brain State : High")
        brainStateLabel.setFixedSize(200, 20)
        accuracyLabel = QLabel("Accuracy of Prediction: 85%")
        accuracyLabel.setFixedSize(400, 20)

        predictionStatisticsVBoxLeft = QVBoxLayout()
        predictionStatisticsVBoxLeft.addWidget(brainStateLabel)

        predictionStatisticsVBoxRight = QVBoxLayout()
        predictionStatisticsVBoxRight.addWidget(accuracyLabel)

        outputHBox.addLayout(predictionStatisticsVBoxLeft)
        outputHBox.addLayout(predictionStatisticsVBoxRight)

        layout.addLayout(outputHBox)
        # layout.addWidget(self.button)
        self.setLayout(layout)

        press_event_id = self.canvas.mpl_connect('button_press_event', self.onclick)
        # motion_event_id = self.canvas.mpl_connect('motion_notify_event', self.on_move)

        self.plot()

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(self, "Select Processed Data Folder")
        if folder_path:
            print(folder_path)
            fNIRLib.loadFile(folder_path, combine=True, points=True)

            return folder_path

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.draw_rectangle(event.xdata, event.ydata)
        self.canvas.draw()

        print(self.features.iloc[0].iloc[0])

    # def on_move(self, event):
    #     # get the x and y pixel coords
    #     x, y = event.x, event.y
    #
    #     if event.inaxes:
    #         ax = event.inaxes  # the axes instance
    #         # self.draw_circle(event.xdata)
    #         print('data coords %f %f' % (event.xdata, event.ydata))
    #     self.figure.tight_layout()

    # def draw_circle(self, x, fig):
    #     circle = plt.Circle((x, 0), .25, **{'color': 'red', 'clip_on': False})
    #     circle = Ellipse((x, 0), .5, .5 * factor,
    #                      **{'color': 'red', 'clip_on': False})
    #     plt.gca().add_artist(circle)
    #     self.canvas.draw()

    def draw_rectangle(self, x, y):
        self.ax.patches.clear()
        rectangle = Rectangle((0, -10), width=x, height=100, color='xkcd:sky blue')
        self.ax.add_patch(rectangle)
        self.canvas.draw()

    def plot(self):
        self.figure.clear()

        # create an axis
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('auto')

        self.ax.plot(self.features[0], '-')
        self.ax.set_xlim(0, 260)
        self.figure.tight_layout()

        # refresh canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
