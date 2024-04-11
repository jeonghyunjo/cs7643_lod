import torch
import torch.nn as nn
import numpy as np

PT_FILE_NAME = "model.pt"

OPTIM_EPOCHS = 3
N_TRIALS = 10
TIMEOUT = 300 # sec

BATCH_SIZE = 32
EPOCHS = 200


# Sample pytorch model 
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 7)  

    def forward(self, x):
        x = x.view(-1, 3) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = torch.mean(x, 0, keepdim=True)
        return x
    

# Sample loss function
loss_fn = nn.SmoothL1Loss()

