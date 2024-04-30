from sklearn.metrics import average_precision_score
import numpy as np
import torch

def compute_ap(outputs, targets, model):
    
    pred_boxes = outputs['pred_boxes']
    
    pred_labels = torch.argmax(outputs['pred_logits'], -1)
    tgt_labels = torch.cat([tgt['labels'] for tgt in targets], dim=0)
    
    APs = {}
    for c in range(model.num_classes):

        mask = tgt_labels == c
        if not mask.any():
            APs[c] = 0.0
            continue
        
        pred_cls = pred_boxes[mask]
        scores = pred_labels[mask] == c
        APs[c] = average_precision_score(scores, pred_cls)
    
    return APs


def evaluate(model, data_loader):
    
    for images, targets in data_loader:
        images = images.to(model.device)
        tgts = [{k: v.to(model.device) for k, v in tgt.items()} for tgt in targets]

        outs = model(images)
        
        APs = compute_ap(outs, tgts)
        
        for cls, ap in APs.items():
            print(f'Class {cls}: AP={ap}')
