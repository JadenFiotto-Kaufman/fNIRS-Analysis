import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QHBoxLayout, QLabel, QAction, QMenuBar, QFileDialog, \
    QTabWidget, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
from sklearn import feature_extraction
from sklearn.externals import joblib
from tsfresh import feature_extraction

from src.main import fNIR
from src.main.fNIRLib import fNIRLib


class Window(QDialog):
    '''
    Initiates the PyQT5 GUI
    '''

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # read in extracted features
        with open('./features_extracted.json', 'r') as file:
            self.loaded_json_data = json.load(file)

        # init lists of widgets for tabs
        self.figures = []
        self.canvases = []

        self.toolbars = []

        # should've made this bottom bar not be per-tab, rather one and its get updated
        self.brainStateLabels = []
        self.pointClickedLabels = []
        self.predictAllButtons = []

        self.axes = []
        self.tabsList = []

        # create menubar
        self.menu_bar = QMenuBar()
        fileMenu = self.menu_bar.addMenu('File')

        # add the Import Data action (Under File)
        importDataAction = QAction('Import Data', self)
        importDataAction.setShortcut('Ctrl+I')
        importDataAction.triggered.connect(self.openFileDialog)

        fileMenu.addAction(importDataAction)

        self.tab_widget = QTabWidget()

        # when you change tabs, spark event
        self.tab_widget.currentChanged.connect(self.onChange)

        # Start the openFileDialog (First thing the user will see)
        self.openFileDialog()

        # if len(self.features) < 1:
        #     #no data imported exit
        #     exit(1)

        # load the model
        self.model = fNIR.load_model()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    def createAllDataTab(self):

        # i = 0 because it is the first tab in all lists
        i = 0

        self.tabsList.append(QWidget())

        # a figure instance to plot on
        self.figures.append(plt.figure())

        # this is the Canvas Widget that displays the plot or (figure)
        self.canvases.append(FigureCanvas(self.figures[i]))

        # this is the Navigation Toolbar for the top of the plot
        self.toolbars.append(NavigationToolbar(self.canvases[i], self))

        # create the layout and menu
        layout = QVBoxLayout()

        layout.setMenuBar(self.menu_bar)
        layout.addWidget(self.toolbars[i])

        layout.addWidget(self.canvases[i])

        outputHBox = QHBoxLayout()

        # create the labels that we will need to access again
        self.brainStateLabels.append(QLabel("Brain State : "))
        self.brainStateLabels[i].setFixedSize(200, 20)

        self.pointClickedLabels.append(QLabel(""))
        self.pointClickedLabels[i].setFixedSize(400, 20)

        # predict on all Button
        predictButton = QPushButton("Predict on All Data", self)
        predictButton.setToolTip('Predicts on all data [0, (Time = Max)]')

        # connect the event to the button when clicked
        predictButton.clicked.connect(self.predictOnAll)
        predictButton.setFixedSize(200, 25)

        self.predictAllButtons.append(predictButton)

        # add all widgets to the layouts, 3 VBox's for 3 Columns
        predictionStatisticsVBoxLeft = QVBoxLayout()
        predictionStatisticsVBoxLeft.addWidget(self.brainStateLabels[i])

        predictionStatisticsVBoxMiddle = QVBoxLayout()
        predictionStatisticsVBoxMiddle.addWidget(self.pointClickedLabels[i])

        predictionStatisticsVBoxRight = QVBoxLayout()
        predictionStatisticsVBoxRight.addWidget(self.predictAllButtons[i])

        # add them all to the bottom HBox (row)
        outputHBox.addLayout(predictionStatisticsVBoxLeft)
        outputHBox.addLayout(predictionStatisticsVBoxMiddle)
        outputHBox.addLayout(predictionStatisticsVBoxRight)

        # set the layout
        layout.addLayout(outputHBox)
        self.tabsList[i].setLayout(layout)

        # add the new tab to the widget
        self.tab_widget.addTab(self.tabsList[i], "All Tasks Data")

        # motion_event_id = self.canvas.mpl_connect('motion_notify_event', self.on_move) #TODO IF A ON_MOUSE_MOVE EVENT IS NEEDED
        # press_event_id = self.canvases[i].mpl_connect('button_press_event', lambda event: self.onclick(event, i))
        # plot the data on the i'th tab
        self.plotAllData()

        self.tabsList[i].setLayout(layout)

    def createTab(self, i):
        '''
        Generates the tab for the task at the index given
        :param i: task number
        '''

        self.tabsList.append(QWidget())

        # a figure instance to plot on
        self.figures.append(plt.figure())

        # this is the Canvas Widget that displays the plot or (figure)
        self.canvases.append(FigureCanvas(self.figures[i]))

        # this is the Navigation Toolbar for the top of the plot
        self.toolbars.append(NavigationToolbar(self.canvases[i], self))

        # create the layout and menu
        layout = QVBoxLayout()

        layout.setMenuBar(self.menu_bar)
        layout.addWidget(self.toolbars[i])

        layout.addWidget(self.canvases[i])

        outputHBox = QHBoxLayout()

        # create the labels that we will need to access again
        self.brainStateLabels.append(QLabel("Brain State : "))
        self.brainStateLabels[i].setFixedSize(200, 20)

        self.pointClickedLabels.append(QLabel(""))
        self.pointClickedLabels[i].setFixedSize(400, 20)

        predictButton = QPushButton("Predict on Task Data", self)
        predictButton.setToolTip('Predicts on all task data')

        predictButton.clicked.connect(self.predictOnAll)
        predictButton.setFixedSize(200, 25)

        self.predictAllButtons.append(predictButton)

        predictionStatisticsVBoxLeft = QVBoxLayout()
        predictionStatisticsVBoxLeft.addWidget(self.brainStateLabels[i])

        predictionStatisticsVBoxMiddle = QVBoxLayout()
        predictionStatisticsVBoxMiddle.addWidget(self.pointClickedLabels[i])

        predictionStatisticsVBoxRight = QVBoxLayout()
        predictionStatisticsVBoxRight.addWidget(self.predictAllButtons[i])

        outputHBox.addLayout(predictionStatisticsVBoxLeft)
        outputHBox.addLayout(predictionStatisticsVBoxMiddle)
        outputHBox.addLayout(predictionStatisticsVBoxRight)

        layout.addLayout(outputHBox)
        self.tabsList[i].setLayout(layout)

        # add the new tab to the widget
        self.tab_widget.addTab(self.tabsList[i], "Task " + str(i))

        # plot the data on the i'th tab
        self.plot(i)

        self.tabsList[i].setLayout(layout)

    def openFileDialog(self):
        '''
        Select the file to import the data from.

        Important function, sets the lay for all tabs

        '''
        options = QFileDialog.Options()

        # return the file_path from the file chosen
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;", options=options)

        # if filepath isnt None, get the data from the file imported
        if file_path:

            # clear all lists (just incase you import more than one file per session)
            self.tab_widget.clear()
            self.figures.clear()
            self.canvases.clear()

            self.toolbars.clear()

            self.brainStateLabels.clear()
            self.pointClickedLabels.clear()
            self.predictAllButtons.clear()

            self.axes.clear()
            self.tabsList.clear()

            # setting the features data
            data = fNIRLib.importSingleton(file_path)

            data_frames = []
            for df in data:
                data_frames.append(df)

            all_features = pd.concat(data_frames)

            # Drop the 16th column (time)
            self.features = all_features.drop(all_features.columns[[16]], axis=1)

            # first create the 'All Data' Tab
            self.createAllDataTab()

            # from 1-20, get the 20 tasks and make new tabs for each
            for i in range(1, int(len(self.features) / 260) + 1):
                self.createTab(i)

            # Set the window title to include the file_path
            self.setWindowTitle("fNIRS Analysis (%s)" % file_path)

    def onclick(self, event, i):
        '''
        Event Listener for when a user clicks on the Plot

        Left Mouse Click:
            Expand the Layout to Tight
            Clear the rectangles off the screen
        Right Mouse Click:
            Draw a rectangle to show where you clicked
            Predict on the data where you clicked
        :param event: The event the user invoked (left mouse button, right mouse button)
        :param i: which tab the user is currently on
        '''
        try:
            # print the data of the event
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
        except TypeError:
            # if you dont click on the plot
            print("Click on Plot")

        # If left mouse button, set tight layout and clear rectangles
        if event.button == 1:
            self.figures[i].tight_layout()
            # clear rectangles
            self.axes[i].patches.clear()

        # If right mouse button, draw rectangle and predict
        if event.button == 3:
            # draw rectangle from event.xdata (Point Clicked) to (i -1) * 260
            self.draw_rectangle(event.xdata, i)

            # round the point clicked
            x_point = int(round(event.xdata))

            # get the points from beginning of task to point clicked
            ml_data = self.features.iloc[(i - 1) * 260:x_point]

            # add the columns needed for tsfresh
            ml_data['column_id'] = ml_data.shape[0] * [0]
            ml_data['time_id'] = range(ml_data.shape[0])

            # load the scaler
            scaler = joblib.load("fNIRscaler.pkl")

            # extract the features
            features_extracted = feature_extraction.extract_features(ml_data,
                                                                     kind_to_fc_parameters=self.loaded_json_data,
                                                                     column_id="column_id",
                                                                     column_sort="time_id")

            features_extracted = pd.DataFrame(scaler.transform(features_extracted), columns=list(features_extracted))

            # predict the class with the features extracted
            predicted_class = self.model.predict_classes(features_extracted)[0][0]

            print(predicted_class)

            # set the labels and update
            if predicted_class == 0:
                self.brainStateLabels[i].setText("Brain State: Low")
            else:
                self.brainStateLabels[i].setText("Brain State: High")

            self.pointClickedLabels[i].setText("Predicted on Time from %s, %s " % ((i - 1) * 260, x_point))
            self.pointClickedLabels[i].update()
            self.brainStateLabels[i].update()

        # update the plot for the rectangle drawn or erased.
        self.canvases[i].draw()

    def onChange(self):
        '''
        If you change tab, set the button press to be activated on current canvas, and set the tab to tight layout
        :return:
        '''
        press_event_id = self.canvases[self.tab_widget.currentIndex()] \
            .mpl_connect('button_press_event',
                         lambda event: self.onclick(event, self.tab_widget.currentIndex()))

        self.figures[self.tab_widget.currentIndex()].tight_layout()  # TODO NOT ALWAYS WORKING

    def draw_rectangle(self, x, i):
        '''
        Draws a rectangle on the current canvas from x = (0 to x)
        :param x: width
        :param i: current tab
        '''
        self.axes[i].patches.clear()
        rectangle = Rectangle((0, -10), width=x, height=100, color='#0F0F0F2F')
        self.axes[i].add_patch(rectangle)
        self.canvases[i].draw()

    def plot(self, i):
        '''
        Plot the data on the current plot
        :param i: tab index
        '''
        # clear figures at i
        self.figures[i].clear()

        # create an axis
        self.axes.append(self.figures[i].add_subplot(111))

        # plot the data for the current task
        self.axes[i].plot(self.features.iloc[(i - 1) * 260: i * 260],
                          '-')  # TODO this may need to be dynamic if the data isnt evenly divisible by 260

        # set x and y labels
        self.axes[i].set_ylabel('Oxygenation or De-Oxygenation Level')
        self.axes[i].set_xlabel('Time Step')

        # create the legend up top with the column names from features
        self.axes[i].legend(self.axes[i].get_lines(), self.features.columns, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=8, mode="expand", borderaxespad=0.)

        # set the x limits for current task
        self.axes[i].set_xlim((i - 1) * 260, i * 260)

        # set tight layout
        self.figures[i].tight_layout()

        # refresh canvas
        self.canvases[i].draw()

    def plotAllData(self):
        '''
        Same as plot(), just from 0 to len(self.features)
        :return:
        '''
        # first tab for All Data (i = 0)
        i = 0

        self.figures[i].clear()

        # create an axis
        self.axes.append(self.figures[i].add_subplot(111))

        # self.axes[i].set_aspect('auto')

        self.axes[i].plot(self.features.iloc[i: len(self.features)],
                          '-')  # TODO this may need to be dynamic if the data isnt evenly divisible by 260

        self.axes[i].set_ylabel('Oxygenation or De-Oxygenation Level')
        self.axes[i].set_xlabel('Time Step')

        self.axes[i].legend(self.axes[i].get_lines(), self.features.columns, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=8, mode="expand", borderaxespad=0.)
        self.axes[i].set_xlim(0, len(self.features))
        self.figures[i].tight_layout()

        # refresh canvas
        self.canvases[i].draw()

    def predictOnAll(self):
        '''
        Predicts on the current task, start of task to the end of the task
        The same as onClick right mouse button.
        :return:
        '''
        i = self.tab_widget.currentIndex()

        ml_data = self.features.iloc[(i - 1) * 260: i * 260]

        ml_data['column_id'] = ml_data.shape[0] * [0]
        ml_data['time_id'] = range(ml_data.shape[0])
        features_extracted = feature_extraction.extract_features(ml_data,
                                                                 kind_to_fc_parameters=self.loaded_json_data,
                                                                 column_id="column_id",
                                                                 column_sort="time_id")
        predicted_class = self.model.predict_classes(features_extracted.values)[0][0]
        if predicted_class == 0:
            self.brainStateLabels[i].setText("Brain State: Low")
        else:
            self.brainStateLabels[i].setText("Brain State: High")

        self.pointClickedLabels[i].setText("Predicted on Data from %s, %s " % ((i - 1) * 260, i * 260))
        self.pointClickedLabels[i].update()
        self.brainStateLabels[i].update()

        self.axes[i].patches.clear()

        self.canvases[i].draw()


# Starts the window for PyQT5
if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
