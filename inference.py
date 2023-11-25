import shutil
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cv2
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from glob import glob
from tqdm import tqdm

id2labels = {0: 1,
            1: 3,
            2: 2,
            3: 4}


def get_submission(video_path, detector_path, 
                   classifier_path, classifier_full_path):
    videos = glob(video_path)
    detector = YOLO(detector_path)
    classifier = YOLO(classifier_path)
    classifier_full = YOLO(classifier_full_path)

    video_names_final = []
    class_names_final = []
    for video in tqdm(videos):
        # cut video
        video_name = video.split('/')[-1].split('.')[0]
        ffmpeg_extract_subclip(
            video, 
            120, 
            150, 
            targetname=f'data/test/{video_name}_temp.mp4'
        )
        
        # get frames
        frames = []
        cap = cv2.VideoCapture(f'data/test/{video_name}_temp.mp4')
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_num = 0
        num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_num % fps == 0:
                    frames.append(frame)
                    num += 1
            else:
                break
            frame_num += 1
        
        cap.release()
        os.remove(f'data/test/{video_name}_temp.mp4')

        if not os.path.exists(f'data/test/{video_name}'):
            os.makedirs(f'data/test/{video_name}')

        if not frames:
            continue

        # yolo detect and save crops
        detector.predict(
            source=frames,
            imgsz=736,
            # save=True,
            save=False,
            conf=0.65,
            iou=0.25,
            # save_crop=False
            save_crop=True,
            project=f'data/test/{video_name}'
        )
        

        # yolo classify cropped frames
        try:
            results = classifier.predict(source=f'data/test/{video_name}/predict/crops/carcase', 
                                        imgsz=384,
                                        save=False           
                                    )  
            
        except FileNotFoundError:
            results = classifier_full.predict(source=frames, 
                                                imgsz=736,
                                                save=False           
                                            )  

        
        shutil.rmtree(f'data/test/{video_name}')
        
        conf = []
        for res in results:
            conf.append(res.probs.data.cpu().numpy())

        conf_mean = list(np.mean(conf, axis=0))

        score = max(conf_mean)
        idx = conf_mean.index(max(conf_mean))
        class_name = id2labels[idx]

        video_names_final.append(video.split('/')[-1])
        class_names_final.append(class_name)

    d = {
        'video_filename': video_names_final,
        'class': class_names_final
    }
    
    df = pd.DataFrame(d)
    df.to_csv('submission.csv', index=False, sep=';')
        


def single_frame_prediction(frame, detector_path, classifier_path):
    detector = YOLO(detector_path)
    classifier = YOLO(classifier_path)

    # yolo detect and save crops
    det_results = detector.predict(
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
    cls_results = classifier.predict(source=f'temp/predict/crops/carcase', 
                            imgsz=384,
                            save=False           
                        )  
    

    shutil.rmtree('temp/predict/')

    scores = list(cls_results[0].probs.data.cpu().numpy())
    score = max(scores)
    idx = scores.index(max(scores))
    class_name = id2labels[idx]

    d = {
        'bbox': bbox,
        'class': class_name,
        'score': score
    }
    
    return d     



if __name__ == "__main__":
    get_submission(video_path='data/test/*',
                   detector_path="services/weights/det.pt",
                   classifier_path="services/weights/cl.pt",
                   classifier_full_path="services/weights/cl_full.pt")
    
    # res = single_frame_prediction(frame='data/yolo_frames/val/Кирпич/Y707AP977_10_17_2023 13_07_59_9.jpg',
    #                               detector_path="services/weights/det.pt",
    #                               classifier_path="services/weights/cl.pt",)
    # print(res)