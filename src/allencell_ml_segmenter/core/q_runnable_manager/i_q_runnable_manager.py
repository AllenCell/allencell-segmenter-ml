from abc import ABC, abstractmethod
from PyQt5.QtCore import QRunnable


class IQRunnableManager(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(self, runnable: QRunnable) -> None:
        pass

    @classmethod
    @abstractmethod
    def global_instance(cls):
        pass
