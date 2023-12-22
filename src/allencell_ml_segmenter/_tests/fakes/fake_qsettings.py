from qtpy.QtCore import QSettings


class FakeQSettings(QSettings):
    def __init__(self):
        self.keys = {}

    def value(self, key):
        return self.keys[key]

    def setValue(self, key, value):
        self.keys[key] = value
