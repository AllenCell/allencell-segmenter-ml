import multiprocessing
import torch


class CUDAUtils:

    @staticmethod
    def cuda_available() -> bool:
        """
        Determines if the user has the correct CUDA drivers installed in order
        accelerate torch using a GPU
        """
        return torch.cuda.is_available()

    @staticmethod
    def get_num_cpu_cores() -> int:
        """
        Get the number of available cpu cores on this machine
        """
        return multiprocessing.cpu_count()
