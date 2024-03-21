from typing import List
from pathlib import Path
from allencell_ml_segmenter.curation.curation_image_loader import (
    CurationImageLoader,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
)
from allencell_ml_segmenter.core.q_runnable_manager import (
    SynchroQRunnableManager,
)
import pytest

raw: List[Path] = [Path("raw 1"), Path("raw 2"), Path("raw 3")]
seg1: List[Path] = [Path("seg1 1"), Path("seg1 2"), Path("seg2 3")]
seg2: List[Path] = [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")]


def test_init_with_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        seg2,
        qr_manager=SynchroQRunnableManager.global_instance(),
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )
    assert loader.get_num_images() == 3
    assert loader.get_raw_image_data() is not None
    assert loader.get_seg1_image_data() is not None
    assert loader.get_seg2_image_data() is not None


def test_init_without_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        qr_manager=SynchroQRunnableManager.global_instance(),
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )
    assert loader.get_num_images() == 3
    assert loader.get_raw_image_data() is not None
    assert loader.get_seg1_image_data() is not None
    assert loader.get_seg2_image_data() is None


def test_init_unequal_lengths():
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            raw,
            seg1,
            [Path("seg2")],
            qr_manager=SynchroQRunnableManager.global_instance(),
            img_data_extractor=FakeImageDataExtractor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            raw,
            [Path("seg1")],
            seg2,
            qr_manager=SynchroQRunnableManager.global_instance(),
            img_data_extractor=FakeImageDataExtractor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [Path("raw")],
            seg1,
            seg2,
            qr_manager=SynchroQRunnableManager.global_instance(),
            img_data_extractor=FakeImageDataExtractor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            raw,
            [Path("seg1")],
            None,
            qr_manager=SynchroQRunnableManager.global_instance(),
            img_data_extractor=FakeImageDataExtractor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [Path("raw")],
            seg1,
            None,
            qr_manager=SynchroQRunnableManager.global_instance(),
            img_data_extractor=FakeImageDataExtractor.global_instance(),
        )


def test_init_empty_lists():
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [],
            [],
            [],
            qr_manager=SynchroQRunnableManager.global_instance(),
            img_data_extractor=FakeImageDataExtractor.global_instance(),
        )
    with pytest.raises(ValueError):
        loader: CurationImageLoader = CurationImageLoader(
            [],
            [],
            None,
            qr_manager=SynchroQRunnableManager.global_instance(),
            img_data_extractor=FakeImageDataExtractor.global_instance(),
        )


def test_next():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        seg2,
        qr_manager=SynchroQRunnableManager.global_instance(),
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )
    assert loader.get_current_index() == 0
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data().path == seg2[0]

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 1
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data().path == seg2[1]

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 2
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data().path == seg2[2]

    assert not loader.has_next()
    with pytest.raises(RuntimeError):
        loader.next()


def test_next_no_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        None,
        qr_manager=SynchroQRunnableManager.global_instance(),
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )
    assert loader.get_current_index() == 0
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data() is None

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 1
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data() is None

    assert loader.has_next()
    loader.next()
    assert loader.get_current_index() == 2
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data() is None

    assert not loader.has_next()
    with pytest.raises(RuntimeError):
        loader.next()


def test_prev():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        seg2,
        qr_manager=SynchroQRunnableManager.global_instance(),
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )
    loader.next()
    loader.next()
    assert loader.get_current_index() == 2
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data().path == seg2[2]

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 1
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data().path == seg2[1]

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 0
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data().path == seg2[0]

    assert not loader.has_prev()
    with pytest.raises(RuntimeError):
        loader.prev()


def test_prev_no_seg2():
    loader: CurationImageLoader = CurationImageLoader(
        raw,
        seg1,
        None,
        qr_manager=SynchroQRunnableManager.global_instance(),
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )
    loader.next()
    loader.next()
    assert loader.get_current_index() == 2
    assert loader.get_raw_image_data().path == raw[2]
    assert loader.get_seg1_image_data().path == seg1[2]
    assert loader.get_seg2_image_data() is None

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 1
    assert loader.get_raw_image_data().path == raw[1]
    assert loader.get_seg1_image_data().path == seg1[1]
    assert loader.get_seg2_image_data() is None

    assert loader.has_prev()
    loader.prev()
    assert loader.get_current_index() == 0
    assert loader.get_raw_image_data().path == raw[0]
    assert loader.get_seg1_image_data().path == seg1[0]
    assert loader.get_seg2_image_data() is None

    assert not loader.has_prev()
    with pytest.raises(RuntimeError):
        loader.prev()
