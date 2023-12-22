from allencell_ml_segmenter._tests.fakes.fake_qsettings import FakeQSettings

from allencell_ml_segmenter.config.user_settings import UserSettings


def test_set_get_user_experiments_path():
    # Arrange
    settings = FakeQSettings()
    userSettings = UserSettings(settings=settings)

    # Act
    userSettings.set_user_experiments_path("foo")

    # Assert
    userSettings.get_user_experiments_path() == "fooo"
