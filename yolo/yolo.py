from ultralytics import YOLO


def run_yolo():
    # Load the model.
    # NOTE: Pick proper YAML file (or load model directly)
    model = YOLO("yolov8-modified.yaml").load("yolov8n.pt")

    # Training
    # NOTE: Hyperparameter tuning here!
    model.train(
        data="yolo_train.yaml",
        imgsz=640,
        epochs=10,
        batch=6,
        name="yolo-ghost-10-epochs",
        device=0,
    )

    # Testing
    metrics = model.val(data="yolo_test.yaml")
    print(metrics.results_dict)


if __name__ == "__main__":
    run_yolo()
