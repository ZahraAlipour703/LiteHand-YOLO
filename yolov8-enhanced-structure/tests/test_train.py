from ultralytics import YOLO


model = YOLO(
    r"D:\zra\PROJECTS\Github\LiteHand-YOLO\yolov8-enhanced-structure\ultralytics\cfg\models\v8\litehand-yolov8n.yaml"
)


model.train(
    data="your_dataset.yaml",
    epochs=1,
    imgsz=640,
    batch=2,
)