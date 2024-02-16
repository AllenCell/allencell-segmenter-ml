from abc import abstractmethod
from qtpy.QtWidgets import QWidget, QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path

from allencell_ml_segmenter.core.subscriber import Subscriber
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from allencell_ml_segmenter.training.metrics_csv_event_handler import MetricsCSVEventHandler

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

class MetricsCSVObserver():
    def __init__(self, csv_path: Path, progress_minimum: int=0, progress_maximum: int=0):
        self._csv_path: Path = csv_path
        self._progress_minimum: int = progress_minimum
        self._progress_maximum: int = progress_maximum
        self._progress: int = progress_minimum

        if not csv_path.exists():
            csv_path.mkdir(parents=True)
        self._target_path: Path = csv_path / f"version_{self._get_last_csv_version() + 1}" / "metrics.csv"
        self._observer: BaseObserver = None

    def get_progress_minimum(self) -> int:
        return self._progress_minimum
    
    def get_progress_maximum(self) -> int:
        return self._progress_maximum
    
    def get_progress(self) -> int:
        return self._progress

    def set_progress(self, progress: int) -> None:
        self._progress = progress if progress >= self._progress_minimum and progress <= self._progress_maximum else self._progress

    def begin_observation_thread(self) -> None:
        self.stop_observation_thread()
        self._observer = Observer()
        event_handler: MetricsCSVEventHandler = MetricsCSVEventHandler(self._target_path, self.set_progress)
        self._observer.schedule(event_handler,  path=self._csv_path,  recursive=True)
        self._observer.start()
    
    def stop_observation_thread(self) -> None:
        if self._observer:
            self._observer.stop()

    def _get_last_csv_version(self) -> int:
        last_version: int = -1
        if self._csv_path.exists():
            for child in self._csv_path.glob("version_*"):
                if child.is_dir():
                    version_str: str = child.name.split("_")[-1]
                    try:
                        last_version = int(version_str) if int(version_str) > last_version else last_version
                    except ValueError:
                        continue
        return last_version
    
class ProgressThread(QThread):
    task_progress: pyqtSignal = pyqtSignal(int)
    def __init__(self, metrics_csv_observer: MetricsCSVObserver, parent=None):
        super(ProgressThread, self).__init__(parent)
        self._metrics_csv_observer: MetricsCSVObserver = metrics_csv_observer

    def run(self):
        while self._metrics_csv_observer.get_progress() < self._metrics_csv_observer.get_progress_maximum():
            self.task_progress.emit(self._metrics_csv_observer.get_progress())
            self.msleep(100)


class View(QWidget, Subscriber, metaclass=ViewMeta):
    """
    Base class for all Views to inherit from
    """

    _template = None

    def __init__(self):
        QWidget.__init__(self)

    def startLongTaskWithProgressBar(self, metrics_csv_observer: MetricsCSVObserver):
        self.longTaskThread = LongTaskThread(do_work=self.doWork)
        self.progressThread = ProgressThread(metrics_csv_observer)
        metrics_csv_observer.begin_observation_thread()

        self.progressDialog = QProgressDialog(
            f"{self.getTypeOfWork()} in Progress", 
            "Cancel", 
            metrics_csv_observer.get_progress_minimum(), 
            metrics_csv_observer.get_progress_maximum(), 
            self
        )
        self.progressDialog.setWindowTitle(f"{self.getTypeOfWork()} Progress")
        self.progressDialog.setWindowModality(Qt.ApplicationModal)
        self.progressDialog.canceled.connect(self.longTaskThread.terminate)
        self.progressDialog.show()

        self.longTaskThread.finished.connect(self.progressDialog.reset)
        self.longTaskThread.finished.connect(self.longTaskThread.deleteLater)
        self.longTaskThread.finished.connect(self.progressDialog.close)
        self.longTaskThread.finished.connect(self.showResults)

        self.progressThread.task_progress.connect(self.updateProgress)
        self.progressThread.finished.connect(metrics_csv_observer.stop_observation_thread)
        self.longTaskThread.finished.connect(metrics_csv_observer.stop_observation_thread)

        
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
