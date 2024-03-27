import torch
class CUDAUtils:

    @staticmethod
    def cuda_available() -> bool:
        """
        Determines if the user has the correct CUDA drivers installed in order
        accelerate torch using a GPU
        """
        return torch.cuda.is_available()
