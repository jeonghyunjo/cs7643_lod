import torch
from ultralytics import YOLO


def train_yolo():
    # Load the model.
    model = YOLO("yolov8n.pt")
    model.train(
        data="yolo.yaml",
        imgsz=640,
        epochs=10,
        batch=8,
        name="yolov8n_traffic",
        device=0,
    )


if __name__ == "__main__":
    train_yolo()
