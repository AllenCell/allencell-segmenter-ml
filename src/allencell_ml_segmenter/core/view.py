from abc import ABC, abstractmethod
from typing import Any
from qtpy.QtWidgets import QWidget, QFrame


class ViewMeta(type(QWidget), type(ABC)):
    pass


class View(ABC, QWidget, metaclass=ViewMeta):
    """
    Base class for all Views to inherit from
    """

    _template = None

    def __init__(self):
        QWidget.__init__(self)


