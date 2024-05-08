from allencell_ml_segmenter.core.task_executor import ITaskExecutor
from typing import Callable, Optional, Any
from napari.qt.threading import FunctionWorker, create_worker


class NapariThreadTaskExecutor(ITaskExecutor):

    _instance = None

    def exec(
        self,
        task: Callable[[], Any],
        on_start: Optional[Callable[[], Any]] = None,
        on_finish: Optional[Callable[[], None]] = None,
        on_return: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        worker: FunctionWorker = create_worker(task)
        if on_start is not None:
            worker.started.connect(on_start)
        if on_finish is not None:
            worker.finished.connect(on_finish)
        if on_return is not None:
            worker.returned.connect(on_return)
        if on_error is not None:
            worker.errored.connect(on_error)
        worker.start()

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
