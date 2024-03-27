from allencell_ml_segmenter.core.q_runnable_manager import IQRunnableManager
from qtpy.QtCore import QRunnable, QThreadPool


class GlobalQRunnableManager(IQRunnableManager):
    """
    Executes QRunnables with the global instance of QThreadPool
    """

    _instance = None

    def run(self, runnable: QRunnable) -> None:
        QThreadPool.globalInstance().start(runnable)

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
