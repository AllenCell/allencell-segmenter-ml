from abc import ABC, abstractmethod
from qtpy.QtWidgets import QWidget, QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class ViewMeta(type(QWidget), type(ABC)):
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

class View(ABC, QWidget, metaclass=ViewMeta):
    """
    Base class for all Views to inherit from
    """

    _template = None

    def __init__(self):
        QWidget.__init__(self)

    def startLongTask(self):
        self.longTaskThread = LongTaskThread(do_work=self.doWork)
        self.progressDialog = QProgressDialog('Long Task in Progress', 'Cancel', 0, 0, self)
        self.progressDialog.setWindowTitle('Progress')
        self.progressDialog.setWindowModality(Qt.ApplicationModal)
        self.progressDialog.canceled.connect(self.longTaskThread.terminate)
        self.progressDialog.show()

        # self.longTaskThread.taskProgress.connect(self.updateProgress)
        self.longTaskThread.finished.connect(self.progressDialog.reset)
        self.longTaskThread.finished.connect(self.longTaskThread.deleteLater)
        self.longTaskThread.finished.connect(self.progressDialog.close)
        self.longTaskThread.start()

    def updateProgress(self, value):
        self.progressDialog.setValue(value)

    @abstractmethod
    def doWork(self):
        pass