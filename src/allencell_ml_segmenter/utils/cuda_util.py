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
    def get_num_workers(use_gpu: bool = False) -> int:
        """
        Get the number of available cpu cores on this machine
        """
        # on mac, we cannot set num_workers no matter what
        if platform.system() == "Darwin":
            return 0
        # on windows/linux, we set num_workers to 1, memory limitations
        # limit this, but num_workers=1 should be able to support most
        # systems while providing a small speed benefit
        else:
            return 1
