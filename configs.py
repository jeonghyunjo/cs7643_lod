import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import models, transforms

import pandas as pd
import os
import math
# import models.detr.models.detr as model
from transformers import DetrConfig, DetrForObjectDetection

config = DetrConfig()
model = DetrForObjectDetection(config)

############## DO NOT CHANGE THE CLASS NAMES AND CONFIG VARIABLE NAMES ##############
class Config:
    def __init__(self):
        self.batch_size = 32                    # Batch size for training
        self.dataset = TLRDataset()           # Dataset
        self.model = model                      # Model
        
        self.hyperparameter_optim = True        # Hyperparameter optimization flag
        self.optimizer = torch.optim.Adam       # Optimizer when hyperparameter_optim is False
        self.optim_epochs = 5                   # Number of epochs of hyperparameter optimizer
        self.optim_n_trials = 100
        self.optim_timeout = 300

        self.lr_min = 0.00001 
        self.lr_max = 0.1
        self.momentum_min = 0.1
        self.momentum_max = 0.9
        self.weight_decay_min = 0.1
        self.weight_decay_max = 0.9

        self.loss_fn = nn.KLDivLoss()           # Loss function
        self.learning_rate = 1e-3               # Learning rate for your optimizer
        self.epochs = 40                        # Number of epochs to train
        self.earlystop_parience = 5             # Number of epochs to wait before early stopping
        self.pt_file_save_path = 'model.pt'     # Path to save pytorch model  
        self.onnx_file_save_path = 'model.onnx' # Path to save onnx model

        # Classification tasks
        self.num_classes = 6                    # Number of classes in the dataset


########## Dataset Setup      ##########    
# Custom dataset preparation for DataLoader
# DO NOT CHANGE THE CLASS NAME
# Sample lidar dataset class
# TODO: Please update this class to provide raw pcd with labels
class TLRDataset(Dataset):
    def __init__(self, dataset_dir, train=True):
        pass
        # H, W = 224, 224
        # self.dir = dataset_dir
        # if train:
        #     self.x_dir = self.dir+'train/'
        # else:
        #     self.x_dir = self.dir+'test/'

        # if train:
        #     df = pd.read_csv(self.dir + 'train.csv')
        # else:
        #     df = pd.read_csv(self.dir + 'test.csv')
            
        # self.img_names = df[df.columns[0]].values.tolist()
        # self.labels = torch.nn.functional.one_hot(torch.as_tensor(df[df.columns[1]].values.tolist()))
        
        # self.to_tensor = v2.Compose([transforms.ToTensor(),
        #                                      transforms.Resize((H,W))])
       
    def __len__(self):
        pass
        # return len(self.img_names)
    
    def __getitem__(self, idx):
        pass
        # x = self.to_tensor(cv2.imread(self.x_dir+'/'+self.img_names[idx]))
        # y = self.labels[idx]
        # return x,y

# TODO: process the raw X input based on our use cases. ex) points -> voxels or ground removal
def preprocessing(raw_X):
    processed_X = raw_X 
    return processed_X
    