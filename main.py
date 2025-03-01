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

import numpy as np

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
print(f"numpy version = {np.__version__}")

from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, kp_remove
# from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
#     jobID_running_myjobs
# from utils import readtxt, writetxt, deleteChineseCharactor, get_subjects, init
# from utils import getMonDate, checkDate, save_nib
# from utils import get_ROIMethod, bar, get_ROIList


def prepareImageNet_data():
    """train
    validation_reorganized
    src/datasets/prepare_ILSVR2012.py"""


def run_LA():
    os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
    kp_run("sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/"
               "config/imagenet_la.json") # 25547786 25547831 25547832 25547836 25547837 25547933 25547938 25547941 25547946


def run_NMPH():
    def torch2numpy():
        os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
        kp_run("sbatch ./NMPH/torch2numpy.sh")  # 25547997

    os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
    kp_run("sbatch ./NMPH/NMPH.sh")


def wholeSet_synapseLevel():
    def run_LA():
        kp_run("sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_la.json")
        # 25553724

    def run_crossEntropy():
        kp_run("sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ft.json")
        # 25553725

    def run_IR():
        kp_run("sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ir.json")
        # 25553726

    def run_LA_layerNorm():
        kp_run("sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_la_layerNorm.json")
        # 25553694 25553721 25553722 25553723


    def run_crossEntropy_layerNorm():
        kp_run("sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ft_layerNorm.json")
        # 25553727

    def run_IR_layerNorm():
        kp_run("sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ir_layerNorm.json")
        # 25553728

    """
    sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_la.json  # 
    sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ft.json  # 25553835 25553837  25553840 25553841
    sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ir.json  # 
    sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_la_layerNorm.json  #  ? 25553773 25553779 25553782  25553786
    sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ft_layerNorm.json  #  25553836
    sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_ir_layerNorm.json  #  25553781 
    
    
    # 25553742 25553743?done 25553744 25553745? 25553746? 25553747?
    # 25553762 25553763? 25553764 25553765? 25553766 25553767
    # 25553770 25553771 25553772 25553773 25553774 25553775
    # 25553788 25553789 25553790 25553791 25553792 25553793
    # 25553795 25553796 25553797 25553798 25553799 25553800
    # 25553828done 25553846done 25553830done 25553831done 25553847done 25553833done
    # rerun ir and ir_norm 25554369 25554370
    """

    def run_NMPH():
        os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
        kp_run("sbatch --array=1-6 ./NMPH/NMPH.sh")
        # 25554106: 25554106_3 25554106_6 is dead
        # imagenet_la p value = 0.039703369140625
        # imagenet_ft p value = 0.07373046875
        # imagenet_la_layer_norm p value = 0.035400390625
        # imagenet_ft_layer_norm p value = 0.084716796875

        # 25554278
        # 25554371


def wholeSet_representationLevel():
    def run_NMPH_representationLevel():
        os.chdir('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch')
        kp_run("sbatch --array=1-6 ./NMPH_representationLevel/NMPH_representationLevel.sh")


def testNMPH_curveFitting():
    pass


def toyData():  # 目的是为了理解表征层次的共激活和表征改变之间的关系.
    f"""NMPH_representationLevel/toyData.py"""


# When loading self.imagenet_dir, how to make sure that this function only loads the specified subfolder name like n01440764 in a text file but not any other subfolder as image classes?
# example text file where the n0... stands for subfolder name: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], "2": ["n01484850", "great_white_shark"], "3": ["n01491361", "tiger_shark"]}

# def load_imagenet_class_index(__json_file_path, __num_classes):
#     with open(__json_file_path, 'r') as f:
#         imagenet_class_index = json.load(f)
#
#     class_labels = [imagenet_class_index[str(i)][0] for i in range(__num_classes)]
#     return class_labels
#
# # Example usage:
# json_file_path = ('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/'
#                   'src/datasets/imagenet_class_index.json')
# num_classes = allowed_subfolders_num
#
# class_labels = load_imagenet_class_index(json_file_path, num_classes)
#
# # Filter subfolders based on the allowed_subfolders list
# if allowed_subfolders_num:
#     subfolders = [folder for folder in os.listdir(self.imagenet_dir) if folder in class_labels]
# else:
#     subfolders = None


def debug_pipeline():
    """
    interactive_gpu
    nvidia-smi
    cd /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/
    . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
    conda activate py36
    python -u /gpfs/milgram/pi/turk-browne/projects/sandbox/sandbox/docker/hello.py

    cd /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch
    python -u ./scripts/instance.py  "/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_la.json"
    # CUDA_VISIBLE_DEVICES=0
    """
    "sbatch ./scripts/instance.sh /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_la.json"
