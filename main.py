"""
this is a file for Kailong to organize things.
to debug: open scripts/instance.py and click on debug
note the interpreter is using docker container: kailongpeng/my_conda_image:v2
    and the interpreter path is /root/miniconda/envs/py36/bin/python
"""

"""
to run the code in dual boot Ubuntu20.04 with docker:
docker run -it kailongpeng/my_conda_image:v2

source activate py36

source init_env.sh

CUDA_VISIBLE_DEVICES=0 python scripts/instance.py ./config/imagenet_ir.json
"""

import os
import sys
import time

import numpy as np
# import pandas as pd

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
print(f"numpy version = {np.__version__}")
# print(f"pandas version = {pd.__version__}")

from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, kp_remove
from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
    jobID_running_myjobs
from utils import readtxt, writetxt, deleteChineseCharactor, get_subjects, init
from utils import getMonDate, checkDate, save_nib
from utils import get_ROIMethod, bar, get_ROIList


def run_LA():
    os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
    kp_run("sbatch ./scripts/instance.sh") # 25547786 25547831 25547832 25547836 25547837 25547933 25547938 25547941 25547946


def run_NMPH():
    def torch2numpy():
        os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
        kp_run("sbatch ./NMPH/torch2numpy.sh")

    os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
    kp_run("sbatch ./NMPH/NMPH.sh")

