import numpy


class ImageProcessing:
    """This class contains utility functions for np.ndarray image data.

    Typical usage example:

      ImageProcessing.set_all_nonzero_values_to(image_array, value_to_set)
    """

    @staticmethod
    def set_all_nonzero_values_to(image: numpy.ndarray, value: int) -> None:
        # set all nonzero values in this :image to :value in-place
        image[image > 0] = value


