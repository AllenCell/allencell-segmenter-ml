import numpy

def set_all_nonzero_values_to(image: numpy.ndarray, value: int) -> None:
    # set all nonzero values in this :image to :value in-place
    image[image > 0] = value
