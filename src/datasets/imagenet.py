"""
Loader for ImageNet. Borrowed heavily from
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import os
import torch.utils.data as data
from torchvision import transforms, datasets

testMode = True
if testMode:
    IMAGENET_DIR = '/opt/project/imagenet/'  # '/home/kp/Desktop/LocalAggregation-Pytorch/imagenet/'  # None
else:
    IMAGENET_DIR = None
DIR_LIST = ['/data5/honglinc/Dataset/imagenet_raw',
            '/data5/chengxuz/Dataset/imagenet_raw',
            '/data5/chengxuz/imagenet_raw',
            '/data/chengxuz/imagenet_raw']

"""
ls -alt /gpfs/milgram/data/imagenet | wc
    ls: 1319444 1319444 26677449
    ls -alt: 1319447 11875016 105844231   There are 1319447 images

ls /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/
    train/: 1000 folders (1000 classes)
    test/: 10000 images
    val/: 50000 images
    
    note
        When I download the dataset, image ID info is not useful to us at that time.
        But I found this info to be useful. The class info is not there because this data is part of a competition.
            https://stackoverflow.com/questions/40744700/how-can-i-find-imagenet-data-labels
        The labels of the 50k val images can be found here: 
            https://github.com/PaddlePaddle/benchmark/blob/master/static_graph/image_classification/pytorch/SENet/ImageData/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
        However, I have not checked whether these labels are correct. I think you can simply check whether they are correct just by eyeballing whether the labels are correct or not. (edited)
        Do you need the test set category too? Since I have not found that just now. If you do need them, we can dig deeper to find it.
"""

for path in DIR_LIST:
    if os.path.exists(path):
        IMAGENET_DIR = path
        break

assert IMAGENET_DIR is not None


class ImageNet(data.Dataset):
    def __init__(self, train=True, imagenet_dir=IMAGENET_DIR, image_transforms=None):
        super().__init__()
        split_dir = 'train' if train else 'validation'
        self.imagenet_dir = os.path.join(imagenet_dir, split_dir)
        self.dataset = datasets.ImageFolder(self.imagenet_dir, image_transforms)

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # important to return the index!
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)
