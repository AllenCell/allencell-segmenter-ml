import multiprocessing
import torch
import platform

class CUDAUtils:

    @staticmethod
    def cuda_available() -> bool:
        """
        Determines if the user has the correct CUDA drivers installed in order
        accelerate torch using a GPU
        """
        return torch.cuda.is_available()

    @staticmethod
    def get_num_workers() -> int:
        """
        Get the number of available cpu cores on this machine
        """
        # on mac, we cannot set num_workers
        if platform.system() == "Darwin":
            return 0
        # on windows/linux, we set num_workers to be number of cores - 1
        # it is recommended to leave one core free
        else:
            return multiprocessing.cpu_count() - 1
