import typing
from qtpy.QtGui import QValidator


# https://doc.qt.io/qtforpython-5/PySide2/QtGui/QValidator.html
class PatchSizeValidator(QValidator):
    # override
    def fixup(self, a0: typing.Optional[str]) -> str:
        """
        This function attempts to change input to be valid according to this validator’s rules.
        It need not result in a valid string: callers of this function must re-test afterwards; the default does nothing
        """
        if a0.isdecimal():
            as_int: int = int(a0)
            # negative and 0 patch sizes not allowed
            if as_int < 4:
                as_int = 4
            # round down to the nearest multiple of 4
            return str(as_int - (as_int % 4))
        else:
            return a0

    # override
    def validate(
        self, a0: typing.Optional[str], a1: int
    ) -> typing.Tuple[QValidator.State, str, int]:
        """
        This virtual function returns Invalid if input is invalid according to this validator’s rules,
        Intermediate if it is likely that a little more editing will make the input acceptable
        (e.g. the user types “4” into a widget which accepts integers between 10 and 99), and Acceptable if the input is valid.

        The function can change both input and pos (the cursor position) if required.
        """
        status: QValidator.State = QValidator.State.Intermediate
        # only if a0 is defined and non-empty can we determine its status
        if a0:
            if a0.isdecimal():
                as_int: int = int(a0)
                if as_int <= 0:
                    status = QValidator.State.Invalid
                elif as_int % 4 == 0:
                    status = QValidator.State.Acceptable
            else:
                status = QValidator.State.Invalid

        return status, a0, a1
