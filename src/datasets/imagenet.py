"""
Loader for ImageNet. Borrowed heavily from
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import os
import torch.utils.data as data
from torchvision import transforms, datasets

testMode = True
if testMode:
    if os.path.exists('/opt/project/imagenet/'):
        IMAGENET_DIR = '/opt/project/imagenet/'  # '/home/kp/Desktop/LocalAggregation-Pytorch/imagenet/'  # None
    else:
        IMAGENET_DIR = '/home/kp/Desktop/LocalAggregation-Pytorch/imagenet/'
else:
    IMAGENET_DIR = None
DIR_LIST = ['/data5/honglinc/Dataset/imagenet_raw',
            '/data5/chengxuz/Dataset/imagenet_raw',
            '/data5/chengxuz/imagenet_raw',
            '/data/chengxuz/imagenet_raw']

"""
cd /gpfs/milgram/data/imagenet/ ; find . -name "*.JPEG" | wc -l
    1,297,600 JPEG images
    cd /gpfs/milgram/data/imagenet/ ; find . -name "*.tar" | wc -l    
    21844 tar files: meaning that there are 21844 classes (e.g. dog, cat, etc.)

ls /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/
    train/: 1,000 folders (1000 classes)
    test/: 100,000 images
    val/: 50,000 images
    
    note
        https://www.image-net.org/download.php
        The most highly-used subset of ImageNet is the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 
        2012-2017 image classification and localization dataset. This dataset spans 1000 object classes and contains 
        1,281,167 training images, 50,000 validation images and 100,000 test images. This subset is available on Kaggle.

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
        # image_data[0].shape = torch.Size([3, 224, 224])
        # image_data[1] = 0
        # important to return the index!
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)
