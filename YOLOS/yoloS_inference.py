from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from torch.utils.data import DataLoader

import torchvision
import torch
from coco_eval import CocoEvaluator
from tqdm import tqdm
import numpy as np
import os
from PIL import Image, ImageDraw

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_cache_dir = "/home/hice1/mwright301/scratch/cs7643_lod/YOLOS/cache"
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", 
                                                cache_dir=custom_cache_dir,
                                                num_labels=1,
                                                ignore_mismatched_sizes=True)
model = model.to(device)


checkpoint_path = "/home/hice1/mwright301/scratch/cs7643_lod/YOLOS/checkpoints/yolos-epoch=13-validation_loss=1.65.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
adjusted_state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items()}
adjusted_state_dict = {key.replace("model.class_labels_classifier", "class_labels_classifier"): value for key, value in adjusted_state_dict.items()}
adjusted_state_dict = {key.replace("model.bbox_predictor", "bbox_predictor"): value for key, value in adjusted_state_dict.items()}
adjusted_state_dict = {key.replace("model.vit.", "vit."): value for key, value in adjusted_state_dict.items()}

#print(checkpoint.keys())  # This will show you the top-level keys in the checkpoint.
#print(checkpoint['state_dict'].keys())  # This will show you the keys under 'state_dict' which should match with your model's parameters.
#print(adjusted_state_dict.keys())
model.load_state_dict(adjusted_state_dict, strict=True)
model.to(device)
# torch.save(model, 'from_ckpt.pt')
model.eval()

# Assuming you use the same processor as the pre-trained model (adjust if not)
fe = AutoFeatureExtractor.from_pretrained("hustvl/yolos-tiny")

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, fe, train=True):
        ann_file = os.path.join(img_folder, "combined_train.json" if train else "combined_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.fe = fe

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.fe(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = fe.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['labels'] = labels
    return batch

batch_size_ = 40
val_dataset = CocoDetection(img_folder='/home/hice1/mwright301/scratch/combined/val', fe=fe, train=False)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size_, num_workers=4)

# initialize evaluator with ground truth (gt)
evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

print("Running evaluation...")
for idx, batch in enumerate(tqdm(val_dataloader)):
    print("Batch ", idx)
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values)

    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = fe.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

    # provide to metric
    # metric expects a list of dictionaries, each item
    # containing image_id, category_id, bbox and score keys
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()

pixel_values, target = val_dataset[2000]

pixel_values = pixel_values.unsqueeze(0).to(device)

with torch.no_grad():
  outputs = model(pixel_values=pixel_values)

import matplotlib.pyplot as plt

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin - (xmax - xmin)/2, ymin - (ymax - ymin)/2), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # plt.show()
    plt.savefig('YOLOS/result_plot.png')
    
# load image based on ID
image_id = target['image_id'].item()
image = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('/home/hice1/mwright301/scratch/combined/val', image['file_name']))

# postprocess model outputs
width, height = image.size
postprocessed_outputs = fe.post_process_object_detection(outputs,
                                                                target_sizes=[(height, width)],
                                                                threshold=0.6)
results = postprocessed_outputs[0]
plot_results(image, results['scores'], results['labels'], results['boxes'])