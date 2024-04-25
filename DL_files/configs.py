import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision.transforms import v2
from torchvision import models, transforms

from PIL import Image, ImageDraw

#import cv2
import pandas as pd
import os
import math
import json

from transformers import DetrConfig, DetrForObjectDetection
import pytorch_lightning as pl

# num_classes = 1
# detr_config = DetrConfig(num_labels=num_classes)
# detr = DetrForObjectDetection(detr_config)


############## DO NOT CHANGE THE CLASS NAMES AND CONFIG VARIABLE NAMES ##############
class Config:
    def __init__(self, dataset_dir):
        self.train_data = CocoDetection(dataset_dir, train=True)   # Dataset
        self.test_data = CocoDetection(dataset_dir, train=False)   # Dataset
        # self.model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)                           # Model
        
        self.bbox_loss_fn = nn.L1Loss()             # Loss function for bounding box
        self.class_loss_fn = nn.BCEWithLogitsLoss()  # Loss function for classification
        self.hyperparameter_optim = False           # Hyperparameter optimization flag
        self.OPTIM_EPOCHS = 2                       # Number of epochs for hyperparameter optimization
        self.N_TRIALS = 100                         # Number of trials for hyperparameter optimization
        self.TIMEOUT = 100                          # Timeout for hyperparameter optimization
        self.lr_min = 0.00001                       # Min learning rate for hyperparameter optimization
        self.lr_max = 0.1                           # Max learning rate for hyperparameter optimization
        self.momentum_min = 0.1
        self.momentum_max = 0.9
        self.weight_decay_min = 0.1
        self.weight_decay_max = 0.9
        self.optimizer = torch.optim.AdamW           # Optimizer when hyperparameter_optim is False
        self.batch_size = 130                         # Batch size for training
        self.learning_rate = 1e-4                   # Learning rate for your optimizer
        self.epochs = 15                            # Number of epochs to train
        self.earlystop_parience = 5                 # Number of epochs to wait before early stopping
        self.pt_file_save_path = 'model.pt'         # Path to save pytorch model  
        self.onnx_file_save_path = 'model.onnx'     # Path to save onnx model

        # Classification tasks
        self.num_classes = 1                    # Number of classes in the dataset


########## Dataset Setup      ##########    
# Custom dataset preparation for DataLoader
# DO NOT CHANGE THE CLASS NAME
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
        
        # Create a set of image_ids that have annotations
        annotated_ids = {ann['image_id'] for ann in self.annotations['annotations']}
        # Filter images list based on the set of annotated_ids
        self.images = [img for img in self.annotations['images'] if img['id'] in annotated_ids]

        # Map annotations to their respective image_id
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
        
        # Input image validation
        # sample_img = cv2.imread(img_path) # For visualization
        # sample_img = cv2.resize(sample_img, (self.new_height, self.new_width)) # For visualization

        if self.transform:
            img = self.transform(img)
        
        annotations = self.id_to_annotations[img_id]
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        # Adjust bounding boxes
        scale_x = self.new_width / original_size[0]
        scale_y = self.new_height / original_size[1]
        scaled_boxes = [[box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y] for box in boxes]

        # Input image validation
        # for box in scaled_boxes:
        #     start_point = (math.ceil(box[0]), math.ceil(box[1]))
        #     end_point = (math.ceil(box[0] + box[2]), math.ceil(box[1] + box[3]))
        #     color = (255,0,0)
        #     thickness = 2
        #     cv2.rectangle(sample_img, start_point, end_point, color, thickness)
        # save_sample_path = "/"
        # cv2.imwrite(save_sample_path + f"/sample_{idx}.png", sample_img)        

        boxes = torch.as_tensor(scaled_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long) 
        target = {
            'boxes': boxes,
            'class_labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return img, target