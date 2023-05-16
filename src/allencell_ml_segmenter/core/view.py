from abc import ABC, abstractmethod
from typing import Any
from qtpy.QtWidgets import QWidget, QFrame


class ViewMeta(type(QWidget), type(ABC)):
    pass


class View(ABC, QWidget, metaclass=ViewMeta):
    """
    Base class for all Views to derive from
    """

    _template = None

    def __init__(self):
        QWidget.__init__(self)

    @abstractmethod
    def load(self, model: Any = None):
        """
        Called when the view is loaded.
        When implementing in child class, use to load initial model data
        and setup the view's UI components

        inputs:
            model - optional model to pass to the view at load time
        """
        pass


class ViewTemplate(View):
    @abstractmethod
    def get_container(self) -> QFrame:
        """
        Get the template's container Frame
        This should be container QFrame in which the child View or ViewTemplate be displayed
        """
        pass

    @abstractmethod
    def load(self):
        """
        Called when the view template is loaded.
        When implementing in child class, use to load initial model data
        and setup the view's UI components
        """
        pass