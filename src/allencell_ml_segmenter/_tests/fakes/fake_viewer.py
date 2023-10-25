from allencell_ml_segmenter.main.i_viewer import IViewer


class FakeViewer(IViewer):
    def __init__(self):
        self._viewer = None

    def add_image(image, name):
        pass
