from abc import abstractmethod
from qtpy.QtWidgets import QWidget, QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.progress_tracker import ProgressTracker


class ViewMeta(type(QWidget), type(Subscriber)):
    pass


class LongTaskThread(QThread):
    taskProgress = pyqtSignal(int)

    def __init__(self, do_work: callable, parent=None):
        super(LongTaskThread, self).__init__(parent)
        self._do_work = do_work

    def run(self):
        print("running")
        # time.sleep(5)
        self._do_work()

        # for i in range(1, 101):
        # self.taskProgress.emit(i)
        # self.msleep(100)  # Simulating some work


class ProgressThread(QThread):
    task_progress: pyqtSignal = pyqtSignal(int)

    def __init__(self, progress_tracker: ProgressTracker, parent=None):
        super(ProgressThread, self).__init__(parent)
        self._progress_tracker: ProgressTracker = progress_tracker

    def run(self):
        while (
            self._progress_tracker.get_progress()
            < self._progress_tracker.get_progress_maximum()
        ):
            self.task_progress.emit(self._progress_tracker.get_progress())
            self.msleep(100)


class View(QWidget, Subscriber, metaclass=ViewMeta):
    """
    Base class for all Views to inherit from
    """

    _template = None

    def __init__(self):
        QWidget.__init__(self)

    def startLongTaskWithProgressBar(self, progress_tracker: ProgressTracker):
        self.longTaskThread = LongTaskThread(do_work=self.doWork)
        self.progressThread = ProgressThread(progress_tracker)

        self.progressDialog = QProgressDialog(
            f"{self.getTypeOfWork()} in Progress",
            "Cancel",
            progress_tracker.get_progress_minimum(),
            progress_tracker.get_progress_maximum(),
            self,
        )
        self.progressDialog.setWindowTitle(f"{self.getTypeOfWork()} Progress")
        self.progressDialog.setWindowModality(Qt.ApplicationModal)
        self.progressDialog.canceled.connect(self.longTaskThread.terminate)
        self.progressDialog.canceled.connect(self.progressThread.terminate)
        self.progressDialog.canceled.connect(progress_tracker.stop_tracker)

        self.progressDialog.show()

        self.longTaskThread.finished.connect(self.progressDialog.reset)
        self.longTaskThread.finished.connect(self.longTaskThread.deleteLater)
        self.longTaskThread.finished.connect(self.progressDialog.close)
        self.longTaskThread.finished.connect(self.showResults)

        self.progressThread.task_progress.connect(self.updateProgress)
        self.progressThread.finished.connect(progress_tracker.stop_tracker)
        self.longTaskThread.finished.connect(progress_tracker.stop_tracker)

        progress_tracker.start_tracker()
        self.progressThread.start()
        self.longTaskThread.start()

    # will remove once prediction is also ported to progress bar
    def startLongTask(self):
        self.longTaskThread = LongTaskThread(do_work=self.doWork)
        self.progressDialog = QProgressDialog(
            f"{self.getTypeOfWork()} in Progress", "Cancel", 0, 0, self
        )
        self.progressDialog.setWindowTitle(f"{self.getTypeOfWork()} Progress")
        self.progressDialog.setWindowModality(Qt.ApplicationModal)
        self.progressDialog.canceled.connect(self.longTaskThread.terminate)
        self.progressDialog.show()

        # self.longTaskThread.taskProgress.connect(self.updateProgress)
        self.longTaskThread.finished.connect(self.progressDialog.reset)
        self.longTaskThread.finished.connect(self.longTaskThread.deleteLater)
        self.longTaskThread.finished.connect(self.progressDialog.close)
        self.longTaskThread.finished.connect(self.showResults)

        self.longTaskThread.start()

    @abstractmethod
    def showResults(self):
        pass

    def updateProgress(self, value):
        self.progressDialog.setValue(value)

    @abstractmethod
    def doWork(self):
        pass

    @abstractmethod
    def getTypeOfWork(self):
        pass
