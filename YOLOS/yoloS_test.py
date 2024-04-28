import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision.transforms import v2
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset

from PIL import Image, ImageDraw

import pandas as pd
import os
import sys
import logging
import math
import json

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import AutoModelForObjectDetection

logPath=os.getcwd()
fileName="torch-training"
rootLogger = logging.getLogger()

num_classes = 6

if len( sys.argv ) != 2:
    rootLogger.error("We need at least ONE command line parameter to run ...")
    rootLogger.error("Exiting now!")
    sys.exit(1)

# Set dataset directory by command line argument
fn = sys.argv[1]
if os.path.exists(fn):
    dataset_dir = fn
    print("Dataset directory: ", dataset_dir)
else:
    print("Dataset directory does not exist.")
    sys.exit(1)

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def custom_collate_fn(batch):
    
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]

    # Stack images because they should have the same dimensions
    images = torch.stack(images, dim=0)

    # Don't stack boxes because they have different dimensions
    # Instead, just pass them as a list
    return (images, boxes)

class CocoDetection(Dataset):
    def __init__(self, dataset_dir, train=True, num_classes=91):
        sub_dir = "train/" if train else "test/"
        json_path = os.path.join(dataset_dir, sub_dir, "annotations.json")
        with open(json_path) as f:
            self.annotations = json.load(f)

        self.img_dir = os.path.join(dataset_dir, sub_dir)
        self.new_height = 480
        self.new_width = 1333
        self.transform = transforms.Compose([
                        transforms.Resize((self.new_height, self.new_width)),
                        transforms.ToTensor(),
                    ])
        
        annotated_ids = {ann['image_id'] for ann in self.annotations['annotations']}
        self.images = [img for img in self.annotations['images'] if img['id'] in annotated_ids]

        self.id_to_annotations = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] in annotated_ids:
                if ann['image_id'] in self.id_to_annotations:
                    self.id_to_annotations[ann['image_id']].append(ann)
                else:
                    self.id_to_annotations[ann['image_id']] = [ann]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        original_size = img.size

        if self.transform:
            img = self.transform(img)
        
        annotations = self.id_to_annotations[img_id]
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        scale_x = self.new_width / original_size[0]
        scale_y = self.new_height / original_size[1]
        scaled_boxes = [[box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y] for box in boxes]

        boxes = torch.as_tensor(scaled_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long) 
        target = {
            'boxes': boxes,
            'class_labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return img, target

train_data = CocoDetection(dataset_dir, train=True)
#train_data = Subset(train_data, indices=range(100))
test_data = CocoDetection(dataset_dir, train=False)
#test_data = Subset(test_data, indices=range(100))
batch_size = 50

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=9)
def save_images_with_boxes(images, annotations, img_dir, scale_x, scale_y, num_images=5):
    for i, img_info in enumerate(images[:num_images]):
        img_id = img_info['id']
        img_path = os.path.join(img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Retrieve annotations for this image
        for ann in annotations[img_id]:
            box = ann['bbox']
            # Scale the bounding box
            box_scaled = [box[0] * scale_x, box[1] * scale_y, (box[0] + box[2]) * scale_x, (box[1] + box[3]) * scale_y]
            # Draw the box
            draw.rectangle(box_scaled, outline="red")
            draw.text((box_scaled[0], box_scaled[1]), str(ann['category_id']), fill="red")
        
        # Save the modified image
        save_path = os.path.join(img_dir, f"annotated_{img_info['file_name']}")
        img.save(save_path)
        print(f"Saved annotated image to {save_path}")

class YoloS(pl.LightningModule):

    def __init__(self, lr, weight_decay):
        super().__init__()
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", 
                                                             num_labels=6,
                                                             ignore_mismatched_sizes=True)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = 50

    def forward(self, images):
        outputs = self.model(pixel_values=images)

        return outputs
     
    def common_step(self, batch):
        pixel_values = batch[0]
        labels = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in batch[1]]
        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return test_dataloader
    
model = YoloS(lr=1e-4, weight_decay=1e-4)

trainer = Trainer(max_epochs=30, gradient_clip_val=0.1)
trainer.fit(model)