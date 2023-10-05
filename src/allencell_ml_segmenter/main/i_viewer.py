from abc import ABC, abstractmethod


class IViewer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def add_image(image, name):
        pass
