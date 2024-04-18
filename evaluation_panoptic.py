from torchvision.models.detection import COCOEvaluator
import torchvision.transforms as T
import torch

def evaluate(model, data_loader, data_set):

    trans = T.Compose([T.ToTensor()])

    eval = COCOEvaluator(data_set, output_dir="./results", eval_panoptic=True)

    model.eval()
    with torch.no_grad():
        for imgs, tgts in data_loader:
            imgs = [trans(img) for img in imgs]
            outputs = model(imgs)
            eval.update(outputs, tgts)

    eval.synchronize_between_processes()
    eval.accumulate()
    eval.summarize()

    APs = eval.coco_eval['panoptic'].eval['precision']
    for cls, ap in APs.items():
        print(f'Class {cls}: AP={ap}')
