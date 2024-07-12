from allencell_ml_segmenter.training.patch_size_validator import (
    PatchSizeValidator,
)
from qtpy.QtGui import QValidator
import pytest


@pytest.fixture
def validator() -> PatchSizeValidator:
    return PatchSizeValidator()


def test_validate_invalid(validator: PatchSizeValidator) -> None:
    assert validator.validate("0", 0)[0] == QValidator.State.Invalid
    assert validator.validate("-10", 0)[0] == QValidator.State.Invalid
    assert validator.validate("f", 0)[0] == QValidator.State.Invalid
    assert validator.validate("some string", 3)[0] == QValidator.State.Invalid
    assert validator.validate("+", 0)[0] == QValidator.State.Invalid
    assert validator.validate("0x", 0)[0] == QValidator.State.Invalid
    assert validator.validate("0b", 0)[0] == QValidator.State.Invalid
    assert validator.validate("0o", 0)[0] == QValidator.State.Invalid


def test_validate_intermediate(validator: PatchSizeValidator) -> None:
    assert validator.validate("1", 0)[0] == QValidator.State.Intermediate
    assert validator.validate("2", 0)[0] == QValidator.State.Intermediate
    assert validator.validate("3", 0)[0] == QValidator.State.Intermediate
    assert validator.validate("22", 0)[0] == QValidator.State.Intermediate
    assert validator.validate("109", 0)[0] == QValidator.State.Intermediate


def test_validate_acceptable(validator: PatchSizeValidator) -> None:
    assert validator.validate("4", 0)[0] == QValidator.State.Acceptable
    assert validator.validate("8", 0)[0] == QValidator.State.Acceptable
    assert validator.validate("64", 0)[0] == QValidator.State.Acceptable
    assert validator.validate("72", 0)[0] == QValidator.State.Acceptable
    assert validator.validate("128", 0)[0] == QValidator.State.Acceptable


def test_fixup(validator: PatchSizeValidator) -> None:
    assert validator.fixup("1") == "4"
    assert validator.fixup("3") == "4"
    assert validator.fixup("5") == "4"
    assert validator.fixup("9") == "8"
    assert validator.fixup("11") == "8"
    assert validator.fixup("131") == "128"
