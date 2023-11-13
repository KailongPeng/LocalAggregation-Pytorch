"""
Loader for ImageNet. Borrowed heavily from
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import json
import os
import torch.utils.data as data
from torchvision import transforms, datasets

testMode = True
if testMode:
    if os.path.exists('/opt/project/imagenet/'):
        IMAGENET_DIR = '/opt/project/imagenet/'  # '/home/kp/Desktop/LocalAggregation-Pytorch/imagenet/'  # None
    elif os.path.exists('/home/kp/Desktop/LocalAggregation-Pytorch/imagenet/'):
        IMAGENET_DIR = '/home/kp/Desktop/LocalAggregation-Pytorch/imagenet/'
    elif os.path.exists("/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/"):
        IMAGENET_DIR = "/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/"
    else:
        raise Exception("imagenet dir not found")
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
        Training data is contained in 1000 folders, one folder per class (each folder should contain 1,300 JPEG images). 
    test/: 100,000 images
    val/: 50,000 images     (128 classes from 1000 classes 50000/1000*128=6400)
    note: create a temporary folder called ./validation/ for debugging, can be safely removed : (how did I created this folder? mkdir validation ; cp -r  train/n031* validation/)
    
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
        
        in this preact-resnet18, the out_dim is set to be 128, which is the number of classes I need in the imagenet dataset.        
"""

for path in DIR_LIST:
    if os.path.exists(path):
        IMAGENET_DIR = path
        break

assert IMAGENET_DIR is not None


class ImageNet(data.Dataset):
    def __init__(self, train=True, imagenet_dir=IMAGENET_DIR, image_transforms=None, allowed_subfolders_num=None):
        super().__init__()
        split_dir = 'train' if train else 'validation_reorganized'
        self.imagenet_dir = os.path.join(imagenet_dir, split_dir)

        # Create the ImageFolder dataset with the filtered subfolders
        dataset = datasets.ImageFolder(self.imagenet_dir, transform=image_transforms)
        self.classes = dataset.classes
        # select the indices of all other folders
        # idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] < allowed_subfolders_num]

        def load_imagenet_class_index(__json_file_path, __num_classes):
            with open(__json_file_path, 'r') as f:
                imagenet_class_index = json.load(f)

            class_labels = [imagenet_class_index[str(i)][0] for i in range(__num_classes)]
            return class_labels

        # Example usage:
        json_file_path = ('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/'
                          'src/datasets/imagenet_class_index.json')
        num_classes = allowed_subfolders_num

        class_labels = load_imagenet_class_index(json_file_path, num_classes)

        idx = [i for i in range(len(dataset)) if dataset.classes[dataset.imgs[i][1]] in class_labels]
        self.classes = class_labels
        # build the appropriate subset

        from torch.utils.data import Subset
        self.dataset = Subset(dataset,
                              idx)  # https://stackoverflow.com/questions/66979537/filter-class-subfolder-with-pytorch-imagefolder
        # print(f"len(self.dataset)={len(self.dataset)}")

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # image_data[0].shape = torch.Size([3, 224, 224])
        # image_data[1] = 0  # image label (ranging from 0-127 when classes num = 128)
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)


# import os
# from torchvision.datasets import VisionDataset
# from typing import Any, Callable, Optional, Tuple
# # import read_image
# # from torchvision.io import read_image
# # from torchvision.io import read_image
# from typing import List, Tuple
# from PIL import Image
# from tqdm import tqdm
# def read_image(path):
#     return Image.open(path).convert('RGB')
#
# class val_CustomImageFolder(VisionDataset):
#     def __init__(
#             self,
#             root: str,
#             labels_folder: str,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             loader: Callable[[str], Any] = read_image,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#     ) -> None:
#         super(val_CustomImageFolder, self).__init__(root, transform=transform,
#                                                     target_transform=target_transform)
#         self.labels_folder = labels_folder
#         self.samples = self._load_samples()
#
#     def extract_name_from_xml(self,
#                               xml_path: str) -> Optional[str]:
#         # xml_path = "/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Annotations/CLS-LOC/val/ILSVRC2012_val_00000001.xml"
#         # class_label = extract_name_from_xml(xml_path)
#         import xml.etree.ElementTree as ET
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#
#         # Find the 'name' tag inside the 'object' tag
#         name_element = root.find(".//object/name")
#
#         # Check if the 'name' tag is found
#         if name_element is not None:
#             # Get the text content inside the 'name' tag
#             name_value = name_element.text
#             return name_value
#         else:
#             # Handle the case where the 'name' tag is not found
#             return None
#
#     def _load_samples(self) -> List[Tuple[str, int]]:
#         labels = []
#         imagePaths = []
#         # Sort the list of filenames
#         sorted_filenames = sorted(os.listdir(self.labels_folder))
#         for curr_val_img in tqdm(sorted_filenames):
#             # Extract the class label from the XML file
#             class_label = self.extract_name_from_xml(f"{self.labels_folder}/{curr_val_img}")
#             labels.append(class_label)
#             image_path = f"{self.root}/{curr_val_img.split('.')[0]}.JPEG"
#             imagePaths.append(image_path)
#         return list(zip(imagePaths, labels))
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return sample, target
#
#     def __len__(self) -> int:
#         return len(self.samples)
#
#     def loader(self, _path):
#         return read_image(_path)
#
#
#
# # Example usage:
# root_folder = "/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/val/"
# labels_file = '/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Annotations/CLS-LOC/val/'  # ILSVRC2012_val_00000001.xml
#
# custom_dataset = val_CustomImageFolder(root=root_folder, labels_folder=labels_file)
