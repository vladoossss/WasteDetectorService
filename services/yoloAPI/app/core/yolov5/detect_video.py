import os
import sys
from pathlib import Path
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO


class YoloPredictor:
    def __init__(
            self, 
            detector_path = '/weights/det.pt', 
            classifier_path = '/weights/cl.pt'
        ) -> None:

        # Load model
        self.detector = YOLO(detector_path)
        self.classifier = YOLO(classifier_path)
        self.id2labels = {
            0: 'Бетон',
            1: 'Грунт',
            2: 'Дерево',
            3: 'Кирпич'
        }

    # @smart_inference_mode()
    def __call__(self, frame):

        det_results = self.detector.predict(
            source=frame,
            imgsz=736,
            # save=True,
            save=False,
            conf=0.65,
            iou=0.25,
            # save_crop=False
            save_crop=True,
            project=f'temp'
        )
        
        if det_results[0].boxes.xyxy.cpu().tolist() == []:
            return None
        

        bbox = list(det_results[0].boxes.xyxy[0].cpu().numpy())
        cls_results = self.classifier.predict(
            source=f'temp/predict/crops/carcase', 
            imgsz=384,
            save=False           
        )  
        

        shutil.rmtree('temp/predict/')

        scores = list(cls_results[0].probs.data.cpu().numpy())
        score = max(scores)
        idx = scores.index(max(scores))
        class_name = self.id2labels[idx]

        d = {
            'bbox': bbox,
            'class_name': class_name,
            'confidence': score
        }
        
        return d     

class YoloBase:
    instance = None
    def __init__(self,) -> None:
        self.yolo = self.create_model()
        YoloBase.instance = self

    def create_model(self):
        return YoloPredictor()

