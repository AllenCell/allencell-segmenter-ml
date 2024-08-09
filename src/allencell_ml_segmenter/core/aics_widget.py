from qtpy.QtWidgets import QWidget

from allencell_ml_segmenter.core.subscriber import Subscriber


class AicsWidgetMeta(type(QWidget), type(Subscriber)):  # type: ignore
    pass


class AicsWidget(QWidget, Subscriber, metaclass=AicsWidgetMeta):
    """
    Base class for custom widgets to inherit from
    """

    _template = None

    def __init__(self) -> None:
        QWidget.__init__(self)
