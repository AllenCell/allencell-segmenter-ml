from abc import abstractmethod, ABC
from qtpy.QtWidgets import QWidget, QProgressDialog
from qtpy.QtCore import Qt, QThread

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

        self.progressDialog = QProgressDialog(
            f"{self.getTypeOfWork()} in Progress",
            "Cancel",
            progress_tracker.get_progress_minimum(),
            progress_tracker.get_progress_maximum(),
            self,
        )
        self.progressDialog.setValue(progress_tracker.get_progress())
        self.progressDialog.setWindowTitle(f"{self.getTypeOfWork()} Progress")
        self.progressDialog.setWindowModality(Qt.ApplicationModal)
        self.progressDialog.canceled.connect(self.longTaskThread.terminate)
        # stop the watchdog thread for file watching inside of the progress tracker
        self.progressDialog.canceled.connect(progress_tracker.stop_tracker)

        self.progressDialog.show()

        self.longTaskThread.finished.connect(self.progressDialog.reset)
        self.longTaskThread.finished.connect(self.longTaskThread.deleteLater)
        self.longTaskThread.finished.connect(self.progressDialog.close)
        self.longTaskThread.finished.connect(self.showResults)

        # connect signals from progress tracker to modify the qprogressdialog
        progress_tracker.signals.progress_changed.connect(self.updateProgress)
        progress_tracker.signals.label_text_changed.connect(
            self.updateLabelText
        )
        progress_tracker.signals.progress_max_changed.connect(
            self.setProgressMax
        )
        # if the longTaskThread or the progressThread finishes, we no longer
        # need to update progress, so we should stop the progress tracker
        self.longTaskThread.finished.connect(progress_tracker.stop_tracker)

        progress_tracker.start_tracker()
        self.longTaskThread.start()

    @abstractmethod
    def showResults(self):
        pass

    def updateProgress(self, value: int) -> None:
        self.progressDialog.setValue(value)

    def updateLabelText(self, value: str) -> None:
        self.progressDialog.setLabelText(value)

    def setProgressMax(self, maximum: int) -> None:
        self.progressDialog.setMaximum(maximum)

    @abstractmethod
    def doWork(self):
        pass

    @abstractmethod
    def getTypeOfWork(self):
        pass


class MainWindow(ABC):
    # this is an ABC that defines a main window in the app, currently this is TrainingView, PredictionView,
    # and CurationMainWidget
    def __init__(self):
        super().__init__()

    @abstractmethod
    def focus_changed(self) -> None:
        pass
