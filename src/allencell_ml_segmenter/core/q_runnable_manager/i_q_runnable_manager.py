from abc import ABC, abstractmethod
from qtpy.QtCore import QRunnable


class IQRunnableManager(ABC):
    def __init__(self):
        raise RuntimeError(
            "Cannot initialize new singleton, please use .global_instance() instead"
        )

    @abstractmethod
    def run(self, runnable: QRunnable) -> None:
        pass

    @classmethod
    @abstractmethod
    def global_instance(cls):
        pass
