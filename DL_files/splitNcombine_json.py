import json
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, test_size=0.2):
    images = data['images']
    annotations = data['annotations']

    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    train_image_ids = set([img['id'] for img in train_images])
    test_image_ids = set([img['id'] for img in test_images])

    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]

    return {"train": {"images": train_images, "annotations": train_annotations},
            "test": {"images": test_images, "annotations": test_annotations}}

def combine_datasets(lisa_split, bosch_split, lisa_categories, bosch_categories):
    max_lisa_cat_id = max(cat['id'] for cat in lisa_categories)
    bosch_categories_adjusted = [{**cat, 'id': cat['id'] + max_lisa_cat_id + 1} for cat in bosch_categories]

    combined_train = {"images": lisa_split['train']['images'] + bosch_split['train']['images'],
                      "annotations": lisa_split['train']['annotations'] + bosch_split['train']['annotations']}
    combined_test = {"images": lisa_split['test']['images'] + bosch_split['test']['images'],
                     "annotations": lisa_split['test']['annotations'] + bosch_split['test']['annotations']}
    combined_categories = lisa_categories + bosch_categories_adjusted
    return combined_train, combined_test, combined_categories

def copy_images(images, src_base, dest_dir):
    for img in images:
        src_path = src_base / img['file_name']
        dest_path = dest_dir / img['file_name']

        # Check if the source file exists before attempting to copy
        if src_path.exists():
            shutil.copy(src_path, dest_path)
        else:
            print(f"Warning: The file {src_path} does not exist and will not be copied.")


def save_data(data, categories, output_path):
    with open(output_path, 'w') as file:
        json.dump({**data, "categories": categories}, file, indent=4)

def main():
    base_dir = Path('/home/christw/Documents/')  # Adjust as needed
    save_base_dir = base_dir
    lisa_path = base_dir / 'trafficlight_dataset_LISA/images/annotations.json'
    bosch_path = base_dir / 'trafficlight_dataset_BOSCH/images/annotations.json'

    lisa_data = load_data(lisa_path)
    bosch_data = load_data(bosch_path)
    lisa_categories = lisa_data['categories']
    bosch_categories = bosch_data['categories']

    lisa_split = split_data(lisa_data)
    bosch_split = split_data(bosch_data)
    combined_train, combined_test, combined_categories = combine_datasets(lisa_split, bosch_split, lisa_categories, bosch_categories)

    train_dir = save_base_dir / 'combined/train'
    test_dir = save_base_dir / 'combined/test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to the new directories
    copy_images(combined_train['images'], base_dir / 'trafficlight_dataset_LISA/images', train_dir)
    copy_images(combined_train['images'], base_dir / 'trafficlight_dataset_BOSCH/images', train_dir)
    copy_images(combined_test['images'], base_dir / 'trafficlight_dataset_LISA/images', test_dir)
    copy_images(combined_test['images'], base_dir / 'trafficlight_dataset_BOSCH/images', test_dir)

    save_data(combined_train, combined_categories, base_dir / 'combined_train.json')
    save_data(combined_test, combined_categories, base_dir / 'combined_test.json')

if __name__ == "__main__":
    main()
