import napari
from magicgui import magicgui
from napari.layers import Layer

from allencell_ml_segmenter.main.segmenter_layer import ImageLayer


@magicgui(call_button='threshold', threshold={"widget_type": "FloatSlider", 'min': .5, 'max': 100})
def simple_thresh_widget(threshold, viewer: napari.Viewer) -> None:
    raw_images: list[Layer] = []
    for l in viewer.layers:
        if l.name.startswith(["raw"]):
            raw_images.append(l)

    for img in raw_images:
        viewer.add_labels((img.data > threshold).astype(int), name=f"[threshold={threshold}] | img.name)")



