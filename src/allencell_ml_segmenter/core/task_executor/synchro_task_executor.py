from allencell_ml_segmenter.core.task_executor import ITaskExecutor
from typing import Callable, Optional, Any


class SynchroTaskExecutor(ITaskExecutor):

    _instance = None
    on_finish: Optional[Callable[[], None]]
    running: bool = False

    def exec(
        self,
        task: Callable[[], Any],
        on_start: Optional[Callable[[], Any]] = None,
        on_finish: Optional[Callable[[], None]] = None,
        on_return: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.running = True
        self.on_finish = on_finish
        if on_start is not None:
            on_start()

        try:
            output: Any = task()
        except Exception as e:
            on_error(e)
            return

        if on_return is not None:
            on_return(output)
        if on_finish is not None:
            on_finish()

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stop_thread(self):
        self.running = False
        self.on_finish()

    def is_worker_running(self) -> bool:
        return self.running
