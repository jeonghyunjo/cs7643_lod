"""
Utility functions related to preparation and formatting of data.
"""

import json
import os
import random
import shutil
import xml.etree.cElementTree as ET
from pathlib import Path

import yaml
from lxml import etree


def delete_labels_and_annotations():
    """
    Delete labels (YOLO) and annotations (Pascal VOC) to reset data generation.
    """

    for stage in ["train", "test", "val"]:
        for format in ["labels", "annotations"]:
            save_dir = Path(__file__).parent.parent / f"data/{format}/{stage}"
            for f in save_dir.glob("*"):
                f.unlink()


def convert_bosch_to_pascal():
    """
    Converts Bosch YAML annotations into Pascal VOC format. Prior to using this function
    ensure the dataset is set up correctly (see README). This function produces bounding
    box annotations for each image as a separate XML file using information in the
    provided YAML files. Note, this function is specific to the Bosch dataset and should
    not be used as a general method of conversion to Pascal VOC.

    The Pascal VOC annotations are stored in the data/annotations directory.

    Adapted from bosch-ros-pkg/bstld GitHub.
    """

    # Constants (specific to Bosch dataset)
    img_width = 1280
    img_height = 720
    img_depth = 3

    # Iterate over train and test data
    for stage in ["train", "test"]:

        # Get directory of YAML and directory to save annotations
        yaml_path = Path(__file__).parent.parent / f"data/{stage}.yaml"
        save_dir = Path(__file__).parent.parent / f"data/annotations/{stage}"

        # Iterate over data for each unique image
        for img_data in yaml.safe_load(open(yaml_path, "r").read()):

            # Begin annotation
            annotation = ET.Element("annotation")

            # Add data about image path
            img_name = str(img_data["path"]).split("/")[-1]
            ET.SubElement(annotation, "folder").text = save_dir.stem
            ET.SubElement(annotation, "filename").text = str(img_name)
            ET.SubElement(annotation, "path").text = str(save_dir / img_name)
            img_name = img_name.split(".")[0]

            # Add image size information
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(img_width)
            ET.SubElement(size, "height").text = str(img_height)
            ET.SubElement(size, "depth").text = str(img_depth)

            # Indicate whether image has been segmented (False)
            ET.SubElement(annotation, "segmented").text = "0"

            # Iterate over bounding boxes
            for box in img_data["boxes"]:
                # Bounding box metadata
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = "Traffic Light"
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "occluded").text = str(box["occluded"])
                ET.SubElement(obj, "difficult").text = "0"

                # Bounding box size information
                bbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bbox, "xmin").text = str(box["x_min"])
                ET.SubElement(bbox, "ymin").text = str(box["y_min"])
                ET.SubElement(bbox, "xmax").text = str(box["x_max"])
                ET.SubElement(bbox, "ymax").text = str(box["y_max"])

            # Save XML file
            # TODO: There is probably a cleaner way to do this
            root = etree.fromstring(ET.tostring(annotation))
            xml_str = etree.tostring(root, pretty_print=True)
            with open(save_dir / (img_name + ".xml"), "wb") as f:
                f.write(xml_str)


def convert_bosch_to_yolo():

    # Constants (specific to Bosch dataset)
    img_width = 1280
    img_height = 720

    # Iterate over train and test data
    for stage in ["train", "test"]:

        # Get directory of YAML and directory to save labels
        yaml_path = Path(__file__).parent.parent / f"data/{stage}.yaml"
        save_dir = Path(__file__).parent.parent / f"data/labels/{stage}"

        # Iterate over data for each unique image
        for img_data in yaml.safe_load(open(yaml_path, "r").read()):
            # Extract image name
            img_name = str(img_data["path"]).split("/")[-1].split(".")[0]

            # Initialize data string
            data_str = ""

            # Iterate over bounding boxes
            for box in img_data["boxes"]:
                # Extract bounding box location
                x_min = float(box["x_min"])
                y_min = float(box["y_min"])
                x_max = float(box["x_max"])
                y_max = float(box["y_max"])

                # Convert to YOLO format
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                x_center = (x_max / img_width) - (width / 2)
                y_center = (y_max / img_height) - (height / 2)

                # Append single line to data string
                traffic_light = 0
                obj_str = f"{traffic_light} {x_center} {y_center} {width} {height}\n"
                data_str += obj_str

            # Save TXT file
            with open(save_dir / (img_name + ".txt"), "w") as f:
                f.write(data_str)


def convert_lisa_to_yolo():
    """
    Run Jeonghyun's lisa_change_structure script first! This script does a nice job of
    extracing all of the images (test & train), merging them, and placing them in a
    single directory. These images should be moved to the train images directory!

    This function parses the generated annotations.json file and formats it in the YOLO
    style.

    The annotations JSON file saves bounding boxes as [x_center, y_center, width, height]
    in absolute image coordinates.
    """

    # Constants (specific to LISA dataset)
    img_width = 1280
    img_height = 960

    # Get annotations filepath
    filepath = Path(__file__).parent.parent / "data/annotations.json"

    # All data moved to train folder
    save_dir = Path(__file__).parent.parent / f"data/labels/train"

    # Iterate though each bounding box
    with open(filepath) as f:
        data = json.load(f)
        yolo_dict = {}
        for annotation in data["annotations"]:
            # Create YOLO formatted bounding box
            bbox = annotation["bbox"]

            # Convert to YOLO format
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            x_center = bbox[0] / img_width
            y_center = bbox[1] / img_height

            # Append single line to string
            traffic_light = 0
            data_str = f"{traffic_light} {x_center} {y_center} {width} {height}\n"

            # Create new string or add to existing
            image_id = annotation["image_id"]
            if image_id in yolo_dict.keys():
                yolo_dict[image_id] = yolo_dict[image_id] + data_str
            else:
                yolo_dict[image_id] = data_str

    # Create TXT files for each entry
    for image_id in range(36264 + 1):
        img_name = f"{image_id}"
        if image_id not in yolo_dict.keys():
            value = ""
        else:
            value = yolo_dict[image_id]
        with open(save_dir / (img_name + ".txt"), "w") as f:
            f.write(value)


def create_validation_dataset(val_split: float = 0.2, random_seed: int = None):
    """
    Create a validation split from the training dataset. The validation data is used to
    tune hyperparameters and assess model performance during training. This is not to be
    confused with the testing dataset (fully withheld during training process). This
    function will reset the data everytime it is called by moving all images and labels
    from the validation directories to the training directory.

    Args:
        val_split (float, optional): Fraction of training data to use for validation
        random_seed (int, optional): Random seed for sampling validation dataset
    """

    # Use random seed
    random.seed(random_seed)

    # Move all validation labels and images back to train directory (reset validation)
    for dir in ["images", "annotations", "labels"]:
        # Define source and destination directories
        source = Path(__file__).parent.parent / f"data/{dir}/val"
        destination = Path(__file__).parent.parent / f"data/{dir}/train"

        # Move files
        for f in os.listdir(source):
            if f != "bosch":
                shutil.copy(source / f, destination / f)
                os.remove(source / f)

    # Randomly sample from the images directory
    train_img_dir = Path(__file__).parent.parent / f"data/images/train"
    files = [f[:-4] for f in os.listdir(train_img_dir) if f != "bosch"]
    val_count = int(len(files) * val_split)
    validation_subset = random.sample(files, val_count)

    # Move each image & label from train to validation directory
    for dir in ["images", "annotations", "labels"]:
        # Define source and destination directories
        source = Path(__file__).parent.parent / f"data/{dir}/train"
        destination = Path(__file__).parent.parent / f"data/{dir}/val"

        # Move files
        for f_name in validation_subset:
            if dir == "images":
                shutil.copy(source / (f_name + ".jpg"), destination / (f_name + ".jpg"))
                os.remove(source / (f_name + ".jpg"))
            # elif dir == "annotations":
            #     shutil.copy(source / (f_name + ".xml"), destination / (f_name + ".xml"))
            #     os.remove(source / (f_name + ".xml"))
            elif dir == "labels":
                shutil.copy(source / (f_name + ".txt"), destination / (f_name + ".txt"))
                os.remove(source / (f_name + ".txt"))


if __name__ == "__main__":
    # delete_labels_and_annotations()
    # convert_bosch_to_pascal()
    # convert_bosch_to_yolo()
    # convert_lisa_to_yolo()
    create_validation_dataset(random_seed=42)
