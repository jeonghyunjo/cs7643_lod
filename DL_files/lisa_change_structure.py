import pandas as pd
import json
import os
import shutil
from PIL import Image
from enum import Enum

class Label(Enum):
    off = 0
    go = 1
    stop = 2
    warning = 3
    goLeft = 4
    stopLeft = 5
    warningLeft = 6
    GreenRight = 7
    RedRight = 8

def find_image_path(image_base, base_paths, filename):
    """ Search for the image file in various base paths and return the full path. """
    for base_path in base_paths:
        potential_path = os.path.join(image_base, base_path, filename)
        if os.path.exists(potential_path):
            # print(potential_path)
            return potential_path
    return None  # If the file is not found

def load_and_prepare_data(annotation_base, image_base, base_paths, destination_folder):
    all_data = pd.DataFrame()
    image_details = {}
    image_id_mapping = {}
    file_id = 0  # Initialize a counter for file IDs
    for root, dirs, files in os.walk(annotation_base):
        for file in files:
            if file.endswith('frameAnnotationsBOX.csv'):
                csv_file_path = os.path.join(root, file)
                data = pd.read_csv(csv_file_path, delimiter=';')
                all_data = pd.concat([all_data, data], ignore_index=True)
                for filename in data['Filename'].unique():
                    pure_filename = os.path.basename(filename)
                    source_path = find_image_path(image_base, base_paths, pure_filename)
                    if source_path:
                        img = Image.open(source_path)
                        image_details[file_id] = {'width': img.width, 'height': img.height}
                        new_filename = f"{file_id}.jpg"
                        img.save(os.path.join(destination_folder, new_filename))  # Save with new ID
                        image_id_mapping[filename] = file_id
                        file_id += 1
                    else:
                        print(f"File not found: {filename}")

    return all_data, image_id_mapping, image_details

def create_coco_json(data, image_id_mapping, image_details):
    images_json = [{
        "id": id,
        "file_name": f"{id}.jpg",
        "width": details['width'],
        "height": details['height']
    } for id, details in image_details.items()]

    annotations_json = []
    annotation_id = 0
    for index, row in data.iterrows():
        filename = row['Filename']
        if filename in image_id_mapping:
            image_id = image_id_mapping[filename]
            try:
                category_id = Label[row['Annotation tag']].value
            except KeyError:
                print(f"Warning: '{row['Annotation tag']}' is not a valid label and will be skipped.")
                continue
            x_center = (row['Upper left corner X'] + row['Lower right corner X']) / 2
            y_center = (row['Upper left corner Y'] + row['Lower right corner Y']) / 2
            width = row['Lower right corner X'] - row['Upper left corner X']
            height = row['Lower right corner Y'] - row['Upper left corner Y']
            bbox = [x_center, y_center, width, height]
            annotations_json.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id += 1
        else:
            print(f"File not found in mapping: {filename}")

    categories_json = [{"id": tag.value, "name": tag.name} for tag in Label]

    return {
        "images": images_json,
        "annotations": annotations_json,
        "categories": categories_json
    }

def save_json(coco_json, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(coco_json, json_file, indent=4)

# Paths configuration
annotation_base = '/home/christw/Documents/trafficlight_dataset_LISA/Annotations/Annotations'
image_base = '/home/christw/Documents/trafficlight_dataset_LISA'
destination_folder = '/home/christw/Documents/trafficlight_dataset_LISA/images'
os.makedirs(destination_folder, exist_ok=True)

json_file_path = 'annotations.json'

base_paths = ['daySequence1/daySequence1/frames',
                  'daySequence2/daySequence2/frames',
                  'dayTrain/dayTrain/dayClip1/frames',
                'dayTrain/dayTrain/dayClip2/frames',
                'dayTrain/dayTrain/dayClip3/frames',
                'dayTrain/dayTrain/dayClip4/frames',
                'dayTrain/dayTrain/dayClip5/frames',
                'dayTrain/dayTrain/dayClip6/frames',
                'dayTrain/dayTrain/dayClip7/frames',
                'dayTrain/dayTrain/dayClip8/frames',
                'dayTrain/dayTrain/dayClip9/frames',
                'dayTrain/dayTrain/dayClip10/frames',
                'dayTrain/dayTrain/dayClip11/frames',
                'dayTrain/dayTrain/dayClip12/frames',
                'dayTrain/dayTrain/dayClip13/frames',
                'nightSequence1/nightSequence1/frames',
                'nightSequence2/nightSequence2/frames',
                'nightTrain/nightTrain/nightClip1/frames',
                'nightTrain/nightTrain/nightClip2/frames',
                'nightTrain/nightTrain/nightClip3/frames',
                'nightTrain/nightTrain/nightClip4/frames',
                'nightTrain/nightTrain/nightClip5/frames']

data, image_id_mapping, image_details = load_and_prepare_data(annotation_base, image_base, base_paths, destination_folder)
coco_json = create_coco_json(data, image_id_mapping, image_details)
save_json(coco_json, json_file_path)

print(f"COCO dataset JSON has been saved to {json_file_path}")


base_file_paths = ['daySequence1/daySequence1/frames',
                  'daySequence2/daySequence2/frames',
                  'dayTrain/dayTrain/dayClip1/frames',
                'dayTrain/dayTrain/dayClip2/frames',
                'dayTrain/dayTrain/dayClip3/frames',
                'dayTrain/dayTrain/dayClip4/frames',
                'dayTrain/dayTrain/dayClip5/frames',
                'dayTrain/dayTrain/dayClip6/frames',
                'dayTrain/dayTrain/dayClip7/frames',
                'dayTrain/dayTrain/dayClip8/frames',
                'dayTrain/dayTrain/dayClip9/frames',
                'dayTrain/dayTrain/dayClip10/frames',
                'dayTrain/dayTrain/dayClip11/frames',
                'dayTrain/dayTrain/dayClip12/frames',
                'dayTrain/dayTrain/dayClip13/frames',
                'nightSequence1/nightSequence1/frames',
                'nightSequence2/nightSequence2/frames',
                'nightTrain/nightTrain/nightClip1/frames',
                'nightTrain/nightTrain/nightClip2/frames',
                'nightTrain/nightTrain/nightClip3/frames',
                'nightTrain/nightTrain/nightClip4/frames',
                'nightTrain/nightTrain/nightClip5/frames']