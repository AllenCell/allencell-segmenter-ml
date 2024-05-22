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
        # For MACOS or CPU runs:
        # On MACOS we cannot set num_workers no matter what.
        # On CPU, increasing num_workers will offer no performance increase
        #   as dataloading is not the bottleneck
        if platform.system() == "Darwin" or not CUDAUtils.cuda_available():
            return 0
        # For Windows/Linux:
        # We set num_workers to 1 for GPU runs
        # num_workers=1 should be able to support most systems while providing a small speed benefit.
        else:
            return 1
