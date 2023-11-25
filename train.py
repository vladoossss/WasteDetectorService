from ultralytics import YOLO


def train_classifier():
    model = YOLO("yolov8s-cls.pt")
    model.train(
        data="data/yolo_frames_filtered", 
        imgsz=384,
        epochs=300,
        batch=256,            
    )  
    
def train_detector():
    model = YOLO("yolov8l.pt")
    model.train(
        data="carcase.yaml", 
        imgsz=736,
        epochs=150,
        batch=16,

        mosaic=0,
        
        patience = 10,
    )  
    
def predict_detector():
    model = YOLO("runs/detect/train_yolo8l/weights/best.pt")
    model.predict(
        source='data/train_frames/Бетон',
        imgsz=736,
        # save=True,
        save=False,
        conf=0.65,
        iou=0.25,
        # save_crop=False
        save_crop=True,
    )

    

if __name__ == "__main__":
    train_detector()
    # predict_detector()
    train_classifier()
    
