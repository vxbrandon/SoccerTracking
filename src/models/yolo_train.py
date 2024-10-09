from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="configs/models/object_detection/yolov11_soccernet.yaml",
                      epochs=5,
                      imgsz=(1920, 1080),
                      project="yolo11_soccernet",
                      batch=2,
                      device='mps',
                      save_period=1,
                      fraction=0.005,
                      plots=True,
                      val=False
                      )

