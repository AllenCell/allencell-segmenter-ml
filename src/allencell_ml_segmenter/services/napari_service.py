import napari

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event


class NapariService(Subscriber):

    def __init__(self, viewer, model):
        super().__init__()
        self.viewer = viewer
        self.model = model

        self.model.subscribe(Event.ACTION_PREDICTION_GRAB_NAPARI_LAYERS, self, self.get_napari_layers)

    def get_napari_layers(self):
        return napari.Viewer.layers
