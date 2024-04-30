import torchvision
import os
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "combined_train.json" if train else "combined_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        #print(target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

fe = AutoFeatureExtractor.from_pretrained("hustvl/yolos-tiny")

train_dataset = CocoDetection(img_folder='/home/hice1/mwright301/scratch/combined/train', feature_extractor=fe)
val_dataset = CocoDetection(img_folder='/home/hice1/mwright301/scratch/combined/val', feature_extractor=fe, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

image_ids = train_dataset.coco.getImgIds()

image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('/home/hice1/mwright301/scratch/combined/train', image['file_name']))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
    box = annotation['bbox']
    print(box)
    class_idx = annotation['category_id']
    x,y,w,h = tuple(box)
    draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
    draw.text((x, y), id2label[class_idx], fill='red')

image.save(f"output_{os.path.basename('/home/hice1/mwright301/scratch/cs7643_lod/YOLOS/sample_output.png')}")

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = fe.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    #batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

batch_size_ = 40
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size_, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size_, num_workers=4)
batch = next(iter(train_dataloader))
batch.keys()

pixel_values, target = train_dataset[0]
pixel_values.shape
print(target)

class YoloS(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        
        self.custom_cache_dir = "/home/hice1/mwright301/scratch/cs7643_lod/YOLOS/cache"
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", 
                                                            cache_dir=self.custom_cache_dir,
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        #pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        self.log("training_loss", loss)
        print("training loss: ", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        print("validation loss: ", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                    weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss',
    dirpath='/home/hice1/mwright301/scratch/cs7643_lod/YOLOS/checkpoints',
    filename='yolos-{epoch:02d}-{validation_loss:.2f}',
    save_top_k=3,
    mode='min',
)

# Setup the EarlyStopping callback
early_stop_callback = EarlyStopping(
    monitor='validation_loss',  # Metric to monitor
    min_delta=0.01,             # Minimum change in the monitored quantity to qualify as an improvement
    patience=10,                # Number of epochs with no improvement after which training will be stopped
    verbose=True,               # Whether to print logs to stdout
    mode='min'                  # Mode 'min' for minimizing the monitored quantity
)

# Setup the trainer with the checkpoint callback
trainer = Trainer(
    max_epochs=25,
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir="/home/hice1/mwright301/scratch/cs7643_lod/YOLOS",
    num_sanity_val_steps=0,
    accelerator='gpu',
)


model = YoloS(lr=1e-4, weight_decay=1e-4)
model = model.to(device)
trainer.fit(model)

# Manually save the final model state
torch.save(model.state_dict(), '/home/hice1/mwright301/scratch/cs7643_lod/YOLOS/final_model.pth')