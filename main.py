# Pytorch related imports
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models, transforms

# Hyperparameter finder modules
import optuna
from optuna.trial import TrialState

# Other modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

import configs
config = configs.Config()

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

# Load dataset
train_data = config.dataset(dataset_dir,train=True)
train_loader = DataLoader(train_data,batch_size=config.batch_size,shuffle=True)

test_data = config.dataset(dataset_dir,train=False)
test_loader = DataLoader(test_data,batch_size=config.batch_size,shuffle=True)


# Load model
model = config.model().to(device)
print(model)

############################################################################################################
# Train function
############################################################################################################
def train(model, optimizer, train_loader):
    train_loss_list = []
    train_acc_list = []
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        # Pre processing on X
        processed_X = configs.preprocessing(X).to(device)

        # Compute prediction error
        pred = model(processed_X)

        # TODO: Check input format for loss function
        loss = config.loss_fn(pred, y.to(torch.float)) 

        # TODO: Set evaluation metric here
        correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            train_loss_list.append(loss)
            train_acc_list.append(correct / len(X))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss_list, train_acc_list

############################################################################################################
# Test function
############################################################################################################
def test(model, test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    cm = torch.zeros(config.num_classes, config.num_classes)
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += config.loss_fn(pred, y.to(torch.float)).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            # update confusion matrix
            _, ys = torch.max(y, 1)
            _, preds = torch.max(pred, 1)
            for t, p in zip(ys.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Handling division by zero
    cm_row_sum = cm.sum(1)
    cm_row_sum[cm_row_sum == 0] = 1  # Prevent division by zero by setting 0s to 1 (or a very small number)
    cm_norm = cm / cm_row_sum[:, None]  # Normalize the confusion matrix

    per_cls_acc = cm_norm.diag().cpu().numpy()  # Extract diagonal (correct predictions) and convert to CPU & NumPy
    for i, acc in enumerate(per_cls_acc):
        print(f"Accuracy of Class {i}: {acc:.4f}")
    
    return correct

############################################################################################################
# Optuna objective function for automatic hyperparameter optimization
############################################################################################################
def objective_optuna(trial):
    init_time = time.time()
    model = model

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


    for epoch in range(config.optim_epochs):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        trial.report(acc, epoch)
        print("Elapsed time: ", time.time() - init_time)
    
    return acc

############################################################################################################
# Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective_optuna, n_trials=config.optim_n_trials, timeout=config.optim_timeout)

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

model = config.model # Init model

if optimizer_name == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=best_params["lr"], 
                                momentum=best_params["momentum"])
elif optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=best_params["lr"], 
                                 betas=(best_params["beta1"], best_params["beta2"]), 
                                 weight_decay=best_params["weight_decay"])

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

plt.figure() 
plt.plot(np.arange(0,len(train_acc_collected))*10,train_acc_collected)
plt.title('Training Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Acc')
plt.savefig('training_accuracy.png')


summary(model, (3, 224, 224))

model_scripted = torch.jit.script(model)
model_scripted.save(config.pt_file_save_path)