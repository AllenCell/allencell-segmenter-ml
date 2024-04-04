from allencell_ml_segmenter.core.q_runnable_manager import IQRunnableManager
from qtpy.QtCore import QRunnable


class SynchroQRunnableManager(IQRunnableManager):
    """
    Executes QRunnables on the calling thread. Used for testing
    """

    _instance = None

    def run(self, runnable: QRunnable) -> None:
        runnable.run()

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
