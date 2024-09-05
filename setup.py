from setuptools import setup
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.call(['python', 'post_install.py'])

setup(
    cmdclass={
        'install': PostInstallCommand,
    },
)