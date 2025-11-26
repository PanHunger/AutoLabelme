from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML

    # Train the model
    results = model.train(data="coco8.yaml", epochs=100, device='1', imgsz=640)