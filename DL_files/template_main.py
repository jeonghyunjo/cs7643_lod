####################### DO NOT CHANGE THIS FILE ########################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import optuna
from optuna.trial import TrialState

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import time


import logging
logPath=os.getcwd()
fileName="torch-training"

# # Set up logging to console AND into file
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# logging.getLogger().setLevel(logging.DEBUG)
rootLogger = logging.getLogger()

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
    return images, boxes

# Load dataset
train_dataset = config.train_data
small_train_dataset = Subset(train_dataset, indices=range(100))
test_dataset = config.test_data
small_test_dataset = Subset(test_dataset, indices=range(100))

train_loader = DataLoader(config.train_data, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(config.test_data, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
small_train_loader = DataLoader(small_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
small_test_loader = DataLoader(small_test_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)


# Load model
testing_model = config.model.to(device)

def convert_bboxes_to_center_format(boxes):
    # Convert from [min_x, min_y, width, height] to [center_x, center_y, width, height]
    center_x = boxes[:, 0] + boxes[:, 2] / 2
    center_y = boxes[:, 1] + boxes[:, 3] / 2
    new_boxes = torch.stack((center_x, center_y, boxes[:, 2], boxes[:, 3]), dim=1)
    return new_boxes

############################################################################################################
# Train function for tuning
############################################################################################################
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

############################################################################################################
# Test function for tuning
############################################################################################################
def test(model, test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    cm = torch.zeros(config.num_classes, config.num_classes)
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in y]
            pred = model(pixel_values=X, labels=y)
            total_loss = pred.loss
            
            # correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            # # update confusion matrix
            # _, ys = torch.max(y, 1)
            # _, preds = torch.max(pred, 1)
            # for t, p in zip(ys.view(-1), preds.view(-1)):
            #     cm[t.long(), p.long()] += 1

    total_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    print(f"Test Error: Avg loss: {total_loss:>8f} \n")

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
study = optuna.create_study(direction="minimize")
study.optimize(objective_optuna, n_trials=config.N_TRIALS, timeout=config.TIMEOUT)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
if trial.value < 0.75:
    print("  Warning: The best trial has low accuracy.")
    sys.exit(-1)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
print("---------------------------------")

############################################################################################################
print("Training the model with the best parameters...")
# Recreate the optimizer with the best parameters
best_params = study.best_trial.params
optimizer_name = best_params.get("optimizer")

model = testing_model # Load model
"""
if optimizer_name == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=best_params["lr"], 
                                momentum=best_params["momentum"])
elif optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=best_params["lr"], 
                                 betas=(best_params["beta1"], best_params["beta2"]), 
                                 weight_decay=best_params["weight_decay"])
"""

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=10e-4, weight_decay=10e-4)

# TODO: Check HuggingFace's optimizer setting first.
# TODO: If they don't have their own optimizer, follow the paper's implementation

train_loss_collected = []
train_acc_collected = []

# Early stopping parameters
best_val_accuracy = 0
patience = 5  # Number of epochs to wait after last improvement
epochs_without_improvement = 0
early_stopping_triggered = False

for t in range(config.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tl, ta = train(model, optimizer, train_loader)
    val_acc = test(model, test_loader)
    
    train_loss_collected.extend(tl)
    train_acc_collected.extend(ta)

    # Check if validation accuracy improved
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        epochs_without_improvement = 0
        # Save your model
        torch.save(model.state_dict(), config.pt_file_save_path)
        # TODO: pt pytorch model file will be saved
        # TODO: make the script to evaluate each pt file
        # For visualization and evaluation purpose.
    else:
        epochs_without_improvement += 1
        
    if epochs_without_improvement == patience:
        print("Early stopping triggered")
        early_stopping_triggered = True
        break
print("Done!")

plt.figure() 
plt.plot(np.arange(0,len(train_loss_collected))*10,train_loss_collected)
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('training_loss.png')

# TODO: modify the accuracy curve or implement another evaluation metric
# TODO: Check HuggingFace's evaluation metric setting first.
plt.figure() 
plt.plot(np.arange(0,len(train_acc_collected))*10,train_acc_collected)
plt.title('Training Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Acc')
plt.savefig('training_accuracy.png')

model_scripted = torch.jit.script(model)
model_scripted.save(config.pt_file_save_path)

# Export the model to ONNX
input_dimension = (1,3,224,224)
input_names = [ "input" ]
output_names = [ "output" ]
sample_model_input = torch.randn(input_dimension).to("cuda:0")
torch.onnx.export(model, 
                  sample_model_input,
                  config.onnx_file_save_path,
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=11,
                  export_params=True,
                  )

