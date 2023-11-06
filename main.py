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
