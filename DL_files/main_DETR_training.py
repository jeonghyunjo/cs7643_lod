import torchvision
import os
from transformers import DetrImageProcessor
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

train_dataset = CocoDetection(img_folder='/home/hice1/jjo49/scratch/combined/train', processor=processor)
val_dataset = CocoDetection(img_folder='/home/hice1/jjo49/scratch/combined/val', processor=processor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

image_ids = train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('/home/hice1/jjo49/scratch/combined/train', image['file_name']))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
    box = annotation['bbox']
    class_idx = annotation['category_id']
    x,y,w,h = tuple(box)
    draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
    draw.text((x, y), id2label[class_idx], fill='red')

image.save(f"output_{os.path.basename('/home/hice1/jjo49/scratch/cs7643_lod/DL_files/sample_output.png')}")

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

batch_size_ = 100
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size_, shuffle=True, num_workers=9)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size_, num_workers=9)
batch = next(iter(train_dataloader))
batch.keys()

pixel_values, target = train_dataset[0]
pixel_values.shape
print(target)

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.custom_cache_dir = "/home/hice1/jjo49/scratch/cs7643_lod/DL_files/cache"
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                                            cache_dir=self.custom_cache_dir,
                                                            revision="no_timm",
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
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
        return val_dataloader

# model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

# outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
# outputs.logits.shape

# trainer = Trainer(max_steps=300, gradient_clip_val=0.1)
# trainer.fit(model)

# model.save_pretrained('/home/hice1/jjo49/scratch/cs7643_lod/DL_files')

# Setup the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss',
    dirpath='/home/hice1/jjo49/scratch/cs7643_lod/DL_files/checkpoints',
    filename='detr-{epoch:02d}-{validation_loss:.2f}',
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
    max_steps=300,
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir="/home/hice1/jjo49/scratch/cs7643_lod/DL_files",
    num_sanity_val_steps=0,
)

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
trainer.fit(model)

# Manually save the final model state
torch.save(model.state_dict(), '/home/hice1/jjo49/scratch/cs7643_lod/DL_files/final_model.pth')