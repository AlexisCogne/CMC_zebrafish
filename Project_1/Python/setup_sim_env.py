"""Clone all FARMS repos"""

import sys
from subprocess import check_call
try:
    from git import Repo
except ImportError:
    check_call([sys.executable, '-m', 'pip', 'install', 'GitPython'])
    from git import Repo
import shutil
import os

def main():
    """Main"""
    pip_install = [sys.executable, '-m', 'pip', 'install']
    pip_uninstall = [sys.executable, '-m', 'pip', 'uninstall']

    # Install MuJoCo
    check_call(pip_install + ['mujoco'])
    check_call(pip_install + ['dm_control'])


    for package in ['farms_core', 'farms_mujoco', 'farms_sim', 'farms_amphibious']:    
        os.chdir(package)
        print(f'Providing option to reinstall {package} if already installed')
        check_call(pip_uninstall+[package])
        check_call(pip_install + ['.'])
        print(f'Completed installation of {package}\n')
        os.chdir("..")


if __name__ == '__main__':
    main()
