from abc import abstractmethod, ABC
from qtpy.QtWidgets import QWidget, QProgressDialog
from qtpy.QtCore import Qt, QThread, Signal

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.progress_tracker import ProgressTracker


class ViewMeta(type(QWidget), type(Subscriber)):
    pass


class LongTaskThread(QThread):
    def __init__(self, do_work: callable, parent=None):
        super().__init__(parent)
        self._do_work = do_work

    # override
    def run(self):
        print("running")
        self._do_work()


class ProgressThread(QThread):
    # pyqtSignal must be class attribute
    # https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#defining-new-signals-with-pyqtsignal
    task_progress: Signal = Signal(int)
    label_text: Signal = Signal(str)

    def __init__(self, progress_tracker: ProgressTracker, parent=None):
        super().__init__(parent)
        self._progress_tracker: ProgressTracker = progress_tracker

    # override
    def run(self):
        while (
            self._progress_tracker.get_progress()
            < self._progress_tracker.get_progress_maximum()
        ):
            self.task_progress.emit(self._progress_tracker.get_progress())
            self.label_text.emit(self._progress_tracker.get_label_text())
            self.msleep(100)


class View(QWidget, Subscriber, metaclass=ViewMeta):
    """
    Base class for all Views to inherit from
    """

    _template = None

    def __init__(self):
        QWidget.__init__(self)

    def startLongTaskWithProgressBar(
        self, progress_tracker: ProgressTracker
    ) -> None:
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
        # stop the watchdog thread for file watching inside of the progress tracker
        self.progressDialog.canceled.connect(progress_tracker.stop_tracker)

        self.progressDialog.show()

        self.longTaskThread.finished.connect(self.progressDialog.reset)
        self.longTaskThread.finished.connect(self.longTaskThread.deleteLater)
        self.longTaskThread.finished.connect(self.progressDialog.close)
        self.longTaskThread.finished.connect(self.showResults)

        # progressThread's task_progress.emit now calls updateProgress
        self.progressThread.task_progress.connect(self.updateProgress)
        self.progressThread.label_text.connect(self.updateLabelText)
        # if the longTaskThread or the progressThread finishes, we no longer
        # need to update progress, so we should stop the progress tracker
        self.progressThread.finished.connect(progress_tracker.stop_tracker)
        self.longTaskThread.finished.connect(progress_tracker.stop_tracker)

        progress_tracker.start_tracker()
        self.progressThread.start()
        self.longTaskThread.start()

    @abstractmethod
    def showResults(self):
        pass

    def updateProgress(self, value: int) -> None:
        self.progressDialog.setValue(value)

    def updateLabelText(self, value: str) -> None:
        self.progressDialog.setLabelText(value)

    @abstractmethod
    def doWork(self):
        pass

    @abstractmethod
    def getTypeOfWork(self):
        pass


class MainWindow(ABC):
    def __init__(self):
        # this is an ABC that defines a main window in the app, currently this is Training, Prediction, and Curation
        super().__init__()

    @abstractmethod
    def focus_changed(self) -> None:
        pass
