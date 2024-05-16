from abc import ABC, abstractmethod
from typing import Callable, Optional, Any


class ITaskExecutor(ABC):
    """
    A TaskExecutor will run the tasks provided to exec() at some point (may be sync or async).
    """

    def __init__(self):
        raise RuntimeError(
            "Cannot initialize new singleton, please use .global_instance() instead"
        )

    @abstractmethod
    def exec(
        self,
        task: Callable[[], Any],
        on_start: Optional[Callable[[], Any]] = None,
        on_finish: Optional[Callable[[], None]] = None,
        on_return: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Execute the provided task. Note that on_return must take the return type of task as a param.
        :param task: task to execute
        :param on_start: runs upon the task starting
        :param on_finish: runs upon the task finishing
        :param on_return: runs upon the task returning, must take return type of task as its only param
        :param on_error: runs upon the task throwing an Exception, must take an Exception as its only param
        """
        pass

    @classmethod
    @abstractmethod
    def global_instance(cls):
        pass
