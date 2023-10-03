# def test_set_patch_size(
#     model_selection_widget: ModelSelectionWidget,
#     training_model: TrainingModel,
# ) -> None:
#     """
#     Tests that using the associated combo box properly sets the patch size field.
#     """

#     for index, patch in enumerate(PatchSize):
#         # ACT
#         model_selection_widget._patch_size_combo_box.setCurrentIndex(index)

#         # ASSERT
#         assert training_model.get_patch_size() == patch


# def test_set_image_dimensions(
#     qtbot: QtBot,
#     model_selection_widget: ModelSelectionWidget,
#     training_model: TrainingModel,
# ) -> None:
#     """
#     Tests that checking the associated radio buttons properly sets the image dimensions.
#     """
#     # ACT
#     with qtbot.waitSignal(model_selection_widget._radio_2d.toggled):
#         model_selection_widget._radio_2d.click()

#     # ASSERT
#     assert training_model.get_image_dims() == 2

#     # ACT
#     with qtbot.waitSignal(model_selection_widget._radio_3d.toggled):
#         model_selection_widget._radio_3d.click()

#     # ASSERT
#     assert training_model.get_image_dims() == 3


# def test_set_max_epoch(
#     qtbot: QtBot,
#     model_selection_widget: ModelSelectionWidget,
#     training_model: TrainingModel,
# ) -> None:
#     """
#     Tests that the max epoch field is properly set by the associated QLineEdit.
#     """
#     # ACT
#     qtbot.keyClicks(model_selection_widget._max_epoch_input, "100")

#     # ASSERT
#     assert training_model.get_max_epoch() == 100


# def test_set_max_time(
#     qtbot: QtBot,
#     model_selection_widget: ModelSelectionWidget,
#     training_model: TrainingModel,
# ) -> None:
#     """
#     Tests that the max time field is properly set by the associated QLineEdit.
#     """
#     # ACT
#     with qtbot.waitSignal(model_selection_widget._max_time_checkbox.toggled):
#         model_selection_widget._max_time_checkbox.click()  # enables the QLineEdit

#     qtbot.keyClicks(model_selection_widget._max_time_in_hours_input, "1")

#     # ASSERT
#     assert training_model.get_max_time() == 3600


# def test_checkbox_slot(
#     qtbot: QtBot, model_selection_widget: ModelSelectionWidget
# ) -> None:
#     """
#     Test the slot connected to the timeout checkbox.
#     """
#     # ASSERT (QLineEdit related to timeout limit is disabled by default)
#     assert not model_selection_widget._max_time_in_hours_input.isEnabled()

#     # ACT (enable QLineEdit related to timeout limit)
#     with qtbot.waitSignal(
#         model_selection_widget._max_time_checkbox.stateChanged
#     ):
#         model_selection_widget._max_time_checkbox.click()

#     # ASSERT
#     assert model_selection_widget._max_time_in_hours_input.isEnabled()

#     # ACT (disabled QLineEdit related to timeout limit)
#     with qtbot.waitSignal(
#         model_selection_widget._max_time_checkbox.stateChanged
#     ):
#         model_selection_widget._max_time_checkbox.click()

#     # ASSERT
#     assert not model_selection_widget._max_time_in_hours_input.isEnabled()
