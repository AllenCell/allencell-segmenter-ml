from abc import ABC, abstractmethod


class IViewer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def add_image(self, image, name):
        pass

    @abstractmethod
    def add_shapes(self, name):
        pass

    @abstractmethod
    def clear_layers(self):
        pass

    @abstractmethod
    def clear_mask_layers(self, layers_to_remove):
        pass

    @abstractmethod
    def get_layers(self):
        pass

    @abstractmethod
    def get_paths_of_image_layers(self):
        pass

    @abstractmethod
    def subscribe_layers_change_event(self, function):
        pass

