from numpy import zeros, ones, ndarray

from allencell_ml_segmenter.utils.image_processing import (
    set_all_nonzero_values_to,
)


def test_set_all_nonzero_values_to():
    # ARRANGE
    array: ndarray = zeros([3, 3, 3])
    array[1, 1, 1] = 255
    array[2, 1, 1] = 3
    array[0, 1, 1] = 230

    # ACT
    array = set_all_nonzero_values_to(array, 2)

    # ASSERT
    assert array.max() == 2
    assert array.shape == (3, 3, 3)
    assert array[1, 1, 1] == 2
    assert array[2, 1, 1] == 2
    assert array[0, 1, 1] == 2
    assert (
        array.sum() == 6
    )  # we set three non-zero values to 2, so we expect 6


def test_set_all_nonzero_values_to_zero_array():
    # ARRANGE
    array: ndarray = zeros([3, 3, 3])

    # ACT
    array = set_all_nonzero_values_to(array, 1)

    # ASSERT
    assert array.max() == 0
    assert array.shape == (3, 3, 3)
    assert (
        array.sum() == 0
    )  # all values in starting array are 0, so nothing should change


def test_set_all_nonzero_values_to_ones_array():
    # ARRANGE
    array: ndarray = ones([3, 3, 3])

    # ACT
    array = set_all_nonzero_values_to(array, 2)

    # ASSERT
    assert array.max() == 2
    assert array.min() == 2
    assert array.shape == (3, 3, 3)
    assert (
        array.sum() == 54
    )  # we set all values in a array of size 27 to 2, so we expect 54
