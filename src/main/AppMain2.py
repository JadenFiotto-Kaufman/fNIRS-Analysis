import sys

import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QHBoxLayout, QLabel, QAction, QMenuBar, QFileDialog, \
    QTabWidget, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from sklearn import feature_extraction

from src.main import fNIR
from src.main.fNIRLib import fNIRLib
from tsfresh import feature_extraction
import json


class Window(QDialog):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        with open('./features_extracted.json', 'r') as file:
            self.loaded_json_data = json.load(file)

        self.figures = []
        self.canvases = []

        self.toolbars = []

        self.brainStateLabels = []
        self.accuracyLabels = []

        self.axes = []
        self.tabsList = []

        menu_bar = QMenuBar()
        fileMenu = menu_bar.addMenu('File')

        importDataAction = QAction('Import Data', self)
        importDataAction.setShortcut('Ctrl+I')
        importDataAction.triggered.connect(self.openFileDialog)

        fileMenu.addAction(importDataAction)

        self.tab_widget = QTabWidget()

        self.tab_widget.currentChanged.connect(self.onChange)

        self.openFileDialog()

        for i in range(0, 20):
            self.tabsList.append(QWidget())

            # a figure instance to plot on
            self.figures.append(plt.figure())

            # this is the Canvas Widget that displays the plot or (figure)
            self.canvases.append(FigureCanvas(self.figures[i]))

            # this is the Navigation Toolbar for the top of the plot
            self.toolbars.append(NavigationToolbar(self.canvases[i], self))

            # create the layout and menu
            layout = QVBoxLayout()

            layout.setMenuBar(menu_bar)
            layout.addWidget(self.toolbars[i])

            layout.addWidget(self.canvases[i])

            outputHBox = QHBoxLayout()

            # create the labels that we will need to access again
            self.brainStateLabels.append(QLabel("Brain State : "))
            self.brainStateLabels[i].setFixedSize(200, 20)
            self.accuracyLabels.append(QLabel("Accuracy of Prediction: "))
            self.accuracyLabels[i].setFixedSize(400, 20)

            predictionStatisticsVBoxLeft = QVBoxLayout()
            predictionStatisticsVBoxLeft.addWidget(self.brainStateLabels[i])

            predictionStatisticsVBoxRight = QVBoxLayout()
            predictionStatisticsVBoxRight.addWidget(self.accuracyLabels[i])

            outputHBox.addLayout(predictionStatisticsVBoxLeft)
            outputHBox.addLayout(predictionStatisticsVBoxRight)

            layout.addLayout(outputHBox)
            self.tabsList[i].setLayout(layout)

            self.tab_widget.addTab(self.tabsList[i], "Task " + str(i + 1))

            # mouse button press event
            # press_event_id = self.canvases[0].mpl_connect('button_press_event', lambda event: self.onclick(event, i))
            # motion_event_id = self.canvas.mpl_connect('motion_notify_event', self.on_move)

            self.plot(i)

            self.tabsList[i].setLayout(layout)

        press_event_id = self.canvases[0].mpl_connect('button_press_event', lambda event: self.onclick(event, i))

        self.model = fNIR.load_model()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

        self.setWindowTitle("fNIRS Analysis")

    def openFileDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        if file_path:
            print(file_path)
            data = fNIRLib.importSingleton(file_path)

            data_frames = []
            for df in data:
                data_frames.append(df)

            all_features = pd.concat(data_frames)
            self.features = all_features.drop(all_features.columns[[16]], axis=1)  # TODO FIX

    def onclick(self, event, i):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        print(str(i))

        if event.button == 1:
            self.figures[i].tight_layout()
            self.axes[i].patches.clear()

        if event.button == 3:
            self.draw_rectangle(event.xdata, i)

            x_point = int(round(event.xdata))

            ml_data = self.features.iloc[0:x_point]  # TODO Figure out math for this
            ml_data['column_id'] = ml_data.shape[0] * [0]
            ml_data['time_id'] = range(ml_data.shape[0])
            print(ml_data)
            features_extracted = feature_extraction.extract_features(ml_data,
                                                                     kind_to_fc_parameters=self.loaded_json_data,
                                                                     column_id="column_id",
                                                                     column_sort="time_id")

            print(features_extracted)

            predicted_class = self.model.predict_classes(features_extracted.values)[0][0]

            if predicted_class == 0:
                self.brainStateLabels[i].setText("Brain State: Low")
            else:
                self.brainStateLabels[i].setText("Brain State: High")

            self.accuracyLabels[i].setText("Prediction Accuracy: 75%")
            self.accuracyLabels[i].update()
            self.brainStateLabels[i].update()
        self.canvases[i].draw()

    def onChange(self):
        press_event_id = self.canvases[self.tab_widget.currentIndex()] \
            .mpl_connect('button_press_event',
                         lambda event: self.onclick(event,self.tab_widget.currentIndex()))
        self.figures[self.tab_widget.currentIndex()].tight_layout()

    # def on_move(self, event):
    #     # get the x and y pixel coords
    #     x, y = event.x, event.y
    #
    #     if event.inaxes:
    #         ax = event.inaxes  # the axes instance
    #         print('data coords %f %f' % (event.xdata, event.ydata))
    #     self.figure.tight_layout()

    def draw_rectangle(self, x, i):
        self.axes[i].patches.clear()
        rectangle = Rectangle((0, -10), width=x, height=100, color='#0F0F0F2F')
        self.axes[i].add_patch(rectangle)
        self.canvases[i].draw()

    def plot(self, i):
        self.figures[i].clear()

        # create an axis
        self.axes.append(self.figures[i].add_subplot(111))

        # self.axes[i].set_aspect('auto')

        self.axes[i].plot(self.features.iloc[i * 260: (i + 1) * 260], '-')

        self.axes[i].legend(self.axes[i].get_lines(), self.features.columns, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=8, mode="expand", borderaxespad=0.)
        self.axes[i].set_xlim(i * 260, (i + 1) * 260)
        self.figures[i].tight_layout()

        # refresh canvas
        self.canvases[i].draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
