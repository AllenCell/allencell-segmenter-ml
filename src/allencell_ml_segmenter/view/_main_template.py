from qtpy.QtWidgets import QFrame, QVBoxLayout, QScrollArea, QLabel
from qtpy.QtCore import Qt

from allencell_ml_segmenter.core.view import ViewTemplate


class MainTemplate(ViewTemplate):
    def __init__(self):
        super().__init__()
        self._container = QFrame()
        self._container.setObjectName("mainContainer")

    def get_container(self) -> QFrame:
        return self._container

    def load(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Page
        page = QFrame()
        page.setObjectName("page")
        page.setLayout(QVBoxLayout())
        page.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)

        # Scroll
        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)  # ScrollBarAsNeeded
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        layout.addWidget(scroll)

        # Container
        self._container.setLayout(QVBoxLayout())
        self._container.layout().setContentsMargins(0, 0, 0, 0)
        page.layout().addWidget(self._container)
        page.layout().addStretch()