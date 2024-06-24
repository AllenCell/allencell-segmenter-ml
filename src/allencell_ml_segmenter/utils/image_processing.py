import numpy


def set_all_nonzero_values_to(
    image: numpy.ndarray, value: int
) -> numpy.ndarray:
    copy = numpy.copy(image)
    # set all nonzero values in this :image to :value in-place
    copy[copy > 0] = value
    return copy
