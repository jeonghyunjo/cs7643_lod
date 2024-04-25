####################### DO NOT CHANGE THIS FILE ########################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import optuna
from optuna.trial import TrialState

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import time
from PIL import Image, ImageDraw
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import evaluate

import logging
logPath=os.getcwd()
fileName="torch-training"

# # Set up logging to console AND into file
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# logging.getLogger().setLevel(logging.DEBUG)
rootLogger = logging.getLogger()

num_classes = 1 #change me

# fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
# fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)

# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

# Check if at least one command line argument was provided
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

# Import configs from configs.py
import configs
config = configs.Config(dataset_dir)

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def custom_collate_fn(batch):
    # Assume each item in batch is a tuple (image, bounding_boxes)
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]

    # Stack images because they should have the same dimensions
    images = torch.stack(images, dim=0)

    # Don't stack boxes because they have different dimensions
    # Instead, just pass them as a list
    return (images, boxes)

# Load dataset
train_dataset = config.train_data
small_train_dataset = Subset(train_dataset, indices=range(100))
test_dataset = config.test_data
small_test_dataset = Subset(test_dataset, indices=range(100))

print("Number of training examples:", len(train_dataset))
print("Number of test examples:", len(test_dataset))

# Check dataset validity before we start to train the model
image_idx = np.random.randint(0, len(train_dataset))
print('Image n°{}'.format(image_idx))

image, annotations = train_dataset[image_idx]
to_pil = transforms.ToPILImage()
image = to_pil(image)

# Create a drawing context
draw = ImageDraw.Draw(image, "RGBA")

# Assuming annotations are in a simplified format, e.g., list of dicts with 'bbox' and 'category'
# print(type(annotations))  # Should output <class 'dict'>
# print(annotations)
# print(image.size)

for idx in range(len(annotations['boxes'])):
    print(annotations['boxes'][idx])
    x, y, w, h = tuple(annotations['boxes'][idx])
    label = annotations['class_labels'][idx]
    draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), outline='red', width=1)
    draw.text((x - w/2, y - h/2), str(label), fill='white') 

# Display the image
image.save("sample_annotated.png")

# Full data loader
train_dataloader = DataLoader(config.train_data, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=9)
test_dataloader = DataLoader(config.test_data, batch_size=config.batch_size, collate_fn=custom_collate_fn, num_workers=4)

# Small data loader
small_train_loader = DataLoader(small_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
small_test_loader = DataLoader(small_test_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.detr_config = DetrConfig(num_labels=1)
        self.model = DetrForObjectDetection(self.detr_config)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, images):
        outputs = self.model(pixel_values=images)

        return outputs

    def common_step(self, batch):
        pixel_values = batch[0]
        # pixel_mask = batch["pixel_mask"]
        # print("Labels data type:", type(batch[1]))
        # print("Labels content:", batch[1])
        labels = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in batch[1]]
        # labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
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
        param_dicts = [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                    weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return test_dataloader

# Load model
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
testing_model = model.to(device)

# batch = next(iter(train_loader))
# batch.keys()
# outputs = testing_model(pixel_values=batch['pixel_values'])
# def convert_bboxes_to_center_format(boxes):
#     # Convert from [min_x, min_y, width, height] to [center_x, center_y, width, height]
#     center_x = boxes[:, 0] + boxes[:, 2] / 2
#     center_y = boxes[:, 1] + boxes[:, 3] / 2
#     new_boxes = torch.stack((center_x, center_y, boxes[:, 2], boxes[:, 3]), dim=1)
#     return new_boxes

def train(model, optimizer, train_loader):
    train_loss_list = []
    train_acc_list = []
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        # labels_list = [{key: y[key][i] for key in y} for i in range(len(y['class_labels']))]
        y = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in y]
        pred = model(pixel_values=X, labels=y)

        total_loss = pred.loss

        # correct = (pred.logits.argmax(1) == y['labels'].argmax(1)).type(torch.float).sum().item()

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            total_loss, current = total_loss.item(), (batch + 1) * len(X)
            train_loss_list.append(total_loss)
            # train_acc_list.append(correct / len(X))
            print(f"loss: {total_loss:>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss_list, train_acc_list

def test(model, test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    cm = torch.zeros(config.num_classes, config.num_classes)
    acc_count = 0
    acc = evaluate.load('accuracy')
    ground_list = []
    pred_list = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in y]
            pred = model(pixel_values=X, labels=y)
            total_loss = pred.loss
            
            ground_list.append(y)
            pred_list.append(pred)

            acc_count += acc.compute(predictions=pred, references=y)
            # correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            # # update confusion matrix
            # _, ys = torch.max(y, 1)
            # _, preds = torch.max(pred, 1)
            # for t, p in zip(ys.view(-1), preds.view(-1)):
            #     cm[t.long(), p.long()] += 1



    total_loss /= num_batches
    avg_acc = acc_count / num_batches

    mean_iou_loader = evaluate.load('mean_iou')
    mean_iou = mean_iou_loader.compute(predictions=pred_list, references=ground_list, num_labels=num_classes, ignore_index=255)

    # correct /= size
    print("Test Error: \n Accuracy: "+ str(100*avg_acc) + "%, Avg loss: " + str(total_loss) + '\n')
    #print(f"Test Error: Avg loss: {total_loss:>8f} \n")
    print("Test Mean IoU: " + str(mean_iou[0]) + '\n')

    # # Handling division by zero
    # cm_row_sum = cm.sum(1)
    # cm_row_sum[cm_row_sum == 0] = 1  # Prevent division by zero by setting 0s to 1 (or a very small number)
    # cm_norm = cm / cm_row_sum[:, None]  # Normalize the confusion matrix

    # per_cls_acc = cm_norm.diag().cpu().numpy()  # Extract diagonal (correct predictions) and convert to CPU & NumPy
    # for i, acc in enumerate(per_cls_acc):
    #     print(f"Accuracy of Class {i}: {acc:.4f}")
    
    return total_loss

############################################################################################################
# Optuna objective function
############################################################################################################
def objective_optuna(trial):
    init_time = time.time()
    model = testing_model

    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    lr = trial.suggest_float("lr", config.lr_min, config.lr_max, log=True)

    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.1, 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", 0.9, 0.999)
        weight_decay = trial.suggest_float("weight_decay", config.weight_decay_min, config.weight_decay_max)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)


    for epoch in range(config.OPTIM_EPOCHS):
        train(model, optimizer, small_train_loader)
        acc = test(model, small_test_loader)
        trial.report(acc, epoch)
        print("Elapsed time: ", time.time() - init_time)
    
    return acc

############################################################################################################
# Optuna study
# study = optuna.create_study(direction="minimize")
# study.optimize(objective_optuna, n_trials=config.N_TRIALS, timeout=config.TIMEOUT)

# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))

# print("Best trial:")
# trial = study.best_trial

# print("  Value: ", trial.value)
# if trial.value < 0.75:
#     print("  Warning: The best trial has low accuracy.")
#     sys.exit(-1)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))
# print("---------------------------------")

############################################################################################################
# print("Training the model with the best parameters...")
print("training started")
# Recreate the optimizer with the best parameters
# best_params = study.best_trial.params
# optimizer_name = best_params.get("optimizer")

model = testing_model # Load model
trainer = Trainer(max_epochs=300, gradient_clip_val=0.1)
trainer.fit(model)

# if optimizer_name == "SGD":
#     optimizer = torch.optim.SGD(model.parameters(), 
#                                 lr=best_params["lr"], 
#                                 momentum=best_params["momentum"])
# elif optimizer_name == "Adam":
#     optimizer = torch.optim.Adam(model.parameters(), 
#                                  lr=best_params["lr"], 
#                                  betas=(best_params["beta1"], best_params["beta2"]), 
#                                  weight_decay=best_params["weight_decay"])

# optimizer = torch.optim.Adam(model.parameters(), 
#                                  lr=1e-4,
#                                  weight_decay=1e-4)
# train_loss_collected = []
# train_acc_collected = []

# # Early stopping parameters
# best_val_accuracy = 0
# patience = 5  # Number of epochs to wait after last improvement
# epochs_without_improvement = 0
# early_stopping_triggered = False

# for t in range(config.epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     tl, ta = train(model, optimizer, train_loader)
#     val_acc = test(model, test_loader)
    
#     train_loss_collected.extend(tl)
#     train_acc_collected.extend(ta)

#     # Check if validation accuracy improved
#     if val_acc > best_val_accuracy:
#         best_val_accuracy = val_acc
#         epochs_without_improvement = 0
#         # Save your model
#         torch.save(model.state_dict(), config.pt_file_save_path)
#     else:
#         epochs_without_improvement += 1
        
#     if epochs_without_improvement == patience:
#         print("Early stopping triggered")
#         early_stopping_triggered = True
#         break
# print("Done!")

# plt.figure() 
# plt.plot(np.arange(0,len(train_loss_collected))*10,train_loss_collected)
# plt.title('Training Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.savefig('training_loss.png')

# plt.figure() 
# plt.plot(np.arange(0,len(train_acc_collected))*10,train_acc_collected)
# plt.title('Training Accuracy')
# plt.xlabel('Iterations')
# plt.ylabel('Acc')
# plt.savefig('training_accuracy.png')

# model_scripted = torch.jit.script(model)
# model_scripted.save(config.pt_file_save_path)

# # Export the model to ONNX
# input_dimension = (1,3,224,224)
# input_names = [ "input" ]
# output_names = [ "output" ]
# sample_model_input = torch.randn(input_dimension).to("cuda:0")
# torch.onnx.export(model, 
#                   sample_model_input,
#                   config.onnx_file_save_path,
#                   verbose=False,
#                   input_names=input_names,
#                   output_names=output_names,
#                   opset_version=11,
#                   export_params=True,
#                   )

