# regroup val set
import json
import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm


def extract_name_from_xml(xml_path):
    # xml_path = "/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Annotations/CLS-LOC/val/ILSVRC2012_val_00000001.xml"
    # class_label = extract_name_from_xml(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the 'name' tag inside the 'object' tag
    name_element = root.find(".//object/name")

    # Check if the 'name' tag is found
    if name_element is not None:
        # Get the text content inside the 'name' tag
        name_value = name_element.text
        return name_value
    else:
        # Handle the case where the 'name' tag is not found
        return None


def move_image_to_class_folder(__image_path, __class_label, __base_folder):
    class_folder = os.path.join(__base_folder, __class_label)

    # Create the class folder if it doesn't exist
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Move the image to the class folder
    shutil.move(__image_path, os.path.join(class_folder, os.path.basename(__image_path)))


for curr_val_img in tqdm(os.listdir("/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/val/")):
    # Path to the JPEG file
    image_path = (f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/val/"
                  f"{curr_val_img.split('.')[0]}.JPEG")

    # Path to the corresponding XML file
    xml_path = (f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Annotations/CLS-LOC/val/"
                f"{curr_val_img.split('.')[0]}.xml")

    # Extract the class label from the XML file
    class_label = extract_name_from_xml(xml_path)

    # Check if the class label is obtained
    if class_label is not None:
        # Base folder for the ImageNet dataset
        base_folder = "/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/validation_reorganized/"

        # Make a new directory named as the class label
        move_image_to_class_folder(image_path, class_label, base_folder)

        # print(f"Image moved to folder: {os.path.join(base_folder, class_label)}")
    else:
        print("Class label not found in the XML file.")


# "/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Annotations/CLS-LOC/val/ILSVRC2012_val_00000001.xml"
# sample 128 classes from the 1000 classes



def moveTo_128():
    # the full folder /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/train is splitted into
    # /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/train and /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/train_128, one has 1000-128=872 classes and the other has 128 classes.
    # the full folder /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/validation_reorganized is splitted into
    # /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/validation_reorganized and /gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/val_128, one has 1000-128=872 classes and the other has 128 classes.

    def load_imagenet_class_index(__json_file_path, __num_classes):
        with open(__json_file_path, 'r') as f:
            imagenet_class_index = json.load(f)

        class_labels = [imagenet_class_index[str(i)][0] for i in range(__num_classes)]
        return class_labels

    # Example usage:
    json_file_path = ('/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/'
                      'src/datasets/imagenet_class_index.json')
    num_classes = 128

    class_labels = load_imagenet_class_index(json_file_path, num_classes)

    for class_label in tqdm(class_labels):
        # move folder f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/validation_reorganized/{class_label}/" to f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/val_128"
        shutil.move(
            f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/validation_reorganized/{class_label}/",
            f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/val_128/{class_label}/")

        # move folder f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/train/{class_label}/" to f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/train_128"
        shutil.move(
            f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/train/{class_label}/",
            f"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Data/CLS-LOC/train_128/{class_label}/")






