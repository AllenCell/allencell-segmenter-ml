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

        self.worker: FunctionWorker = create_worker(task)
        if on_start is not None:
            self.worker.started.connect(on_start)
        if on_finish is not None:
            self.worker.finished.connect(on_finish)
        if on_return is not None:
            self.worker.returned.connect(on_return)
        if on_error is not None:
            self.worker.errored.connect(on_error)
        self.worker.start()

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stop_thread(self) -> None:
        # Ask all workers to quit, and wait for them to do so.
        self.worker.await_workers()

    def is_worker_running(self) -> bool:
        return self.worker.is_running

