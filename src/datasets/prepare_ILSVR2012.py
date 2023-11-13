# regroup val set
import xml.etree.ElementTree as ET


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




"/gpfs/milgram/project/turk-browne/projects/localize/ImageNet/ILSVRC/Annotations/CLS-LOC/val/ILSVRC2012_val_00000001.xml"
# sample 128 classes from the 1000 classes

