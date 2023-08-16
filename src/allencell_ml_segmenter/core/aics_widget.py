from abc import ABC
from qtpy.QtWidgets import QWidget

class AicsWidgetMeta(type(QWidget), type(ABC)):
    pass

class AicsWidget(ABC, QWidget, metaclass=AicsWidgetMeta):
    """
    Base class for custom widgets to inherit from
    """

    _template = None

    def __init__(self):
        QWidget.__init__(self)
