import yaml
import shutil
from pathlib import Path
import os
import json
from PIL import Image  # Importing the Python Image Library
from enum import Enum

class Label(Enum):
    off = 0
    Green = 1
    Red = 2
    Yellow = 3
    GreenLeft = 4
    RedLeft = 5
    YellowLeft = 6
    GreenRight = 7
    RedRight = 8

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def ensure_dir(path):
    """Ensure that a directory exists, and if not, create it."""
    os.makedirs(path, exist_ok=True)

def process_images(data, image_num, images, annotations, image_dir, image_output_dir):
    for item in data:
        image_id = image_num
        image_num += 1
        original_path = image_dir / Path(item['path']).name
        filename = f"{image_id}.jpg"
        
        # Copy the image to the single output directory with a new filename
        new_image_path = image_output_dir / filename
        shutil.copy(original_path, new_image_path)

        # Load the image to get its dimensions
        with Image.open(original_path) as img:
            width, height = img.size

        # Add image info to images list
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # Add annotation info to annotations list
        for box in item['boxes']:
            if box['x_min'] < 0 or box['y_min'] < 0 or box['x_max'] > width or box['y_max'] > height:
                print("Warning: Box coordinates out of bounds: ", image_id)
                continue
            try:
                annotations.append({
                    "id": len(annotations),
                    "image_id": image_id,
                    "category_id": Label[box['label']].value,
                    "bbox": [(box['x_min'] + box['x_max'])/2, 
                             (box['y_min'] + box['y_max'])/2, 
                             box['x_max'] - box['x_min'], 
                             box['y_max'] - box['y_min']],
                    "area": (box['x_max'] - box['x_min']) * (box['y_max'] - box['y_min']),
                    "iscrowd": 0
                })
            except KeyError:
                continue

    return image_num

def main():
    base_dir = Path('/home/christw/Documents/trafficlight_dataset_BOSCH')
    image_output_dir = base_dir / 'images'  # Single folder for all images
    ensure_dir(image_output_dir)  # Ensure the single image directory exists
    
    train_data = load_yaml(base_dir / 'train.yaml')
    test_data = load_yaml(base_dir / 'test.yaml')
    train_image_dir = base_dir / 'rgb' / 'train'
    test_image_dir = base_dir / 'rgb' / 'test'

    images = []
    annotations = []
    image_num = 36265  # Start numbering from 36265

    # Process each dataset part
    image_num = process_images(train_data, image_num, images, annotations, train_image_dir, image_output_dir)
    process_images(test_data, image_num, images, annotations, test_image_dir, image_output_dir)

    # Generate categories from the Enum
    categories = [{"id": label.value, "name": label.name} for label in Label]

    # Compile the entire dataset into one JSON
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to JSON file
    with open(base_dir / 'annotations.json', 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    main()
