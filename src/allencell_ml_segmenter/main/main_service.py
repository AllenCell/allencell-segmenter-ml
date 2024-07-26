from allencell_ml_segmenter.main.main_model import MainModel, ImageType
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.core.task_executor import (
    ITaskExecutor,
    NapariThreadTaskExecutor,
)
from allencell_ml_segmenter.utils.file_writer import IFileWriter, FileWriter
import json
from pathlib import Path
from typing import Optional
from allencell_ml_segmenter.core.event import Event


class MainService:
    def __init__(
        self,
        main_model: MainModel,
        experiments_model: IExperimentsModel,
        task_executor: ITaskExecutor = NapariThreadTaskExecutor.global_instance(),
        file_writer: IFileWriter = FileWriter.global_instance(),
    ):
        self._main_model: MainModel = main_model
        self._experiments_model: IExperimentsModel = experiments_model
        self._task_executor: ITaskExecutor = task_executor
        self._file_writer: IFileWriter = file_writer

        self._main_model.signals.selected_channels_changed.connect(
            self._write_selected_channels
        )
        # TODO: refactor experiments model to use slots + signals
        self._experiments_model.subscribe(
            Event.ACTION_EXPERIMENT_APPLIED, self, self._read_selected_channels
        )

    def _read_selected_channels(self, e: Event) -> None:
        channel_path: Optional[Path] = (
            self._experiments_model.get_channel_selection_path()
        )
        if channel_path is not None:
            self._task_executor.exec(
                lambda: self._read_channel_json(channel_path),
                on_return=self._main_model.set_selected_channels,
            )

    def _read_channel_json(
        self, channel_path: Path
    ) -> dict[ImageType, Optional[int]]:
        if channel_path.exists():
            with open(channel_path, "r") as fr:
                channels: dict[str, Optional[int]] = json.load(fr)
            typed_channels: dict[ImageType, Optional[int]] = {
                ImageType(k): v for k, v in channels.items()
            }
            return typed_channels
        return {
            ImageType.RAW: None,
            ImageType.SEG1: None,
            ImageType.SEG2: None,
        }

    def _write_selected_channels(self) -> None:
        selected_channels: dict[ImageType, Optional[int]] = (
            self._main_model.get_selected_channels()
        )
        # this is a non-critical task, so failing silently is OK--user will just have to manually specify
        # channels during training
        self._task_executor.exec(
            lambda: self._write_channel_json(selected_channels)
        )

    def _write_channel_json(
        self, selected_channels: dict[ImageType, Optional[int]]
    ) -> None:
        jsonified_channels: dict[str, Optional[int]] = {
            k.value: v for k, v in selected_channels.items()
        }
        self._file_writer.write_json(
            jsonified_channels,
            self._experiments_model.get_channel_selection_path(),
        )
