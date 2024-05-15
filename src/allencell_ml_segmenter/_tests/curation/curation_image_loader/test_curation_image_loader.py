from typing import List
from pathlib import Path
from allencell_ml_segmenter.curation.curation_image_loader import (
    CurationImageLoader,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
)
from allencell_ml_segmenter.core.task_executor import SynchroTaskExecutor
import pytest
from unittest.mock import Mock

raw: List[Path] = [Path("raw 1"), Path("raw 2"), Path("raw 3")]
seg1: List[Path] = [Path("seg1 1"), Path("seg1 2"), Path("seg2 3")]
seg2: List[Path] = [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")]


def test_init_unequal_lengths():
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            raw,
            seg1,
            [Path("seg2")],
            img_data_extractor=FakeImageDataExtractor.global_instance(),
            task_executor=SynchroTaskExecutor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            raw,
            [Path("seg1")],
            seg2,
            img_data_extractor=FakeImageDataExtractor.global_instance(),
            task_executor=SynchroTaskExecutor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [Path("raw")],
            seg1,
            seg2,
            img_data_extractor=FakeImageDataExtractor.global_instance(),
            task_executor=SynchroTaskExecutor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            raw,
            [Path("seg1")],
            None,
            img_data_extractor=FakeImageDataExtractor.global_instance(),
            task_executor=SynchroTaskExecutor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [Path("raw")],
            seg1,
            None,
            img_data_extractor=FakeImageDataExtractor.global_instance(),
            task_executor=SynchroTaskExecutor.global_instance(),
        )


def test_init_empty_lists():
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [],
            [],
            [],
            img_data_extractor=FakeImageDataExtractor.global_instance(),
            task_executor=SynchroTaskExecutor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [],
            [],
            None,
            img_data_extractor=FakeImageDataExtractor.global_instance(),
            task_executor=SynchroTaskExecutor.global_instance(),
        )


def test_start_with_seg2():
    # arrange
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        seg2,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        task_executor=SynchroTaskExecutor.global_instance(),
    )

    on_first_ready_mock: Mock = Mock()
    loader.signals.first_image_ready.connect(on_first_ready_mock)
    on_next_ready_mock: Mock = Mock()
    loader.signals.next_image_ready.connect(on_next_ready_mock)

    # act
    loader.start()

    # assert
    assert loader.get_num_images() == 3 # num paths in our file lists (raw, seg1, seg2)
    on_first_ready_mock.assert_called_once()
    on_next_ready_mock.assert_called_once()
    assert loader.get_raw_image_data() is not None
    assert loader.get_seg1_image_data() is not None
    assert loader.get_seg2_image_data() is not None


def test_start_without_seg2():
    # arrange
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        task_executor=SynchroTaskExecutor.global_instance(),
    )

    on_first_ready_mock: Mock = Mock()
    loader.signals.first_image_ready.connect(on_first_ready_mock)
    on_next_ready_mock: Mock = Mock()
    loader.signals.next_image_ready.connect(on_next_ready_mock)

    # act
    loader.start()

    # assert
    assert loader.get_num_images() == 3 # num paths in our file lists (raw, seg1, seg2)
    on_first_ready_mock.assert_called_once()
    on_next_ready_mock.assert_called_once()
    assert loader.get_raw_image_data() is not None
    assert loader.get_seg1_image_data() is not None
    assert loader.get_seg2_image_data() is None


def test_next_with_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        seg2,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        task_executor=SynchroTaskExecutor.global_instance(),
    )
    on_next_ready_mock: Mock = Mock()
    loader.signals.next_image_ready.connect(on_next_ready_mock)
    loader.start()

    assert loader.get_current_index() == 0
    assert on_next_ready_mock.call_count == 1
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data().path == seg2[0]

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 1
    assert on_next_ready_mock.call_count == 2
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data().path == seg2[1]

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 2
    assert (
        on_next_ready_mock.call_count == 2
    )  # there is no next to load, so the signal shouldn't be emitted
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data().path == seg2[2]

    assert not loader.has_next()
    with pytest.raises(RuntimeError):
        loader.next()


def test_next_without_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        None,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        task_executor=SynchroTaskExecutor.global_instance(),
    )
    on_next_ready_mock: Mock = Mock()
    loader.signals.next_image_ready.connect(on_next_ready_mock)
    loader.start()

    assert loader.get_current_index() == 0
    assert on_next_ready_mock.call_count == 1
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data() is None

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 1
    assert on_next_ready_mock.call_count == 2
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data() is None

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 2
    assert on_next_ready_mock.call_count == 2
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data() is None

    assert not loader.has_next()
    with pytest.raises(RuntimeError):
        loader.next()


def test_prev_with_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        seg2,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        task_executor=SynchroTaskExecutor.global_instance(),
    )
    on_prev_ready_mock: Mock = Mock()
    loader.signals.prev_image_ready.connect(on_prev_ready_mock)
    loader.start()
    loader.next()
    loader.next()

    assert loader.get_current_index() == 2
    assert on_prev_ready_mock.call_count == 0
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data().path == seg2[2]

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 1
    assert on_prev_ready_mock.call_count == 1
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data().path == seg2[1]

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 0
    assert on_prev_ready_mock.call_count == 1
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data().path == seg2[0]

    assert not loader.has_prev()
    with pytest.raises(RuntimeError):
        loader.prev()


def test_prev_without_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        None,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        task_executor=SynchroTaskExecutor.global_instance(),
    )
    on_prev_ready_mock: Mock = Mock()
    loader.signals.prev_image_ready.connect(on_prev_ready_mock)
    loader.start()
    loader.next()
    loader.next()

    assert loader.get_current_index() == 2
    assert on_prev_ready_mock.call_count == 0
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data() is None

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 1
    assert on_prev_ready_mock.call_count == 1
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data() is None

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 0
    assert on_prev_ready_mock.call_count == 1
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data() is None

    assert not loader.has_prev()
    with pytest.raises(RuntimeError):
        loader.prev()
