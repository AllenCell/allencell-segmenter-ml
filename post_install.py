import sys
import subprocess

def install_torch():
    if sys.platform.startswith('win') or sys.platform.startswith('linux'):
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torch', "torchvision"])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==2.0.1+cu118', "torchvision" "--index-url", "https://download.pytorch.org/whl/cu118"])

if __name__ == '__main__':
    install_torch()