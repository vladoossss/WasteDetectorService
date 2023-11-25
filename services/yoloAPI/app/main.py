from typing import OrderedDict
from app.core.yolov5.detect_video import YoloBase
from app.utils.helpers import plot_one_box
from app.core.yolov5.object_tracking.object_tracking import ObjectTracker
from collections import OrderedDict, defaultdict

from fastapi import FastAPI

import cv2
import gradio as gr

GRADIO_PATH = ""

app = FastAPI(title='API')

yolo_model = YoloBase()
    
def gradio_predict(file):

    last_prediction = 'cap'
    
    cap = cv2.VideoCapture(file)

    h, m, s = map(int, '00:02:05'.split(":"))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip_id = (((h * 60) + m) * 60) * fps
    cap.set(propId=cv2.CAP_PROP_POS_FRAMES, value=int(frame_skip_id))
    
    if cap is None or not cap.isOpened():
        return None, None

    detected_objects = []
    total_frames = 0

    tracker = ObjectTracker(
        skip_frames = 1,
        min_tracked_frames_for_drawing = 0,
        max_disappeared_frames = 5,
        max_tracked_distance = 600,
    )
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer.open('/tmp/tmp.mp4', fourcc, fps, (frame_width, frame_height), True)


    res = defaultdict(float)

    cnt=0
    while cap.isOpened():
        ret, original_rgb_frame = cap.read() 
        cnt+=1
        if not ret or cnt > 15 * fps:
            break

        original_rgb_frame = cv2.cvtColor(original_rgb_frame, cv2.COLOR_BGR2RGB)

        detected_objects.clear()
        
        prediction = yolo_model.yolo(original_rgb_frame)
        # формат prediction: 
        # 
        # {
        #     "class_name": self.names[int(result[5])],
        #     "bbox": [int(x) for x in result[:4].tolist()], #convert bbox results to int from float
        #     "confidence": float(result[4]),
        # } 
        # 

        if prediction is not None:
            detected_objects.append(prediction)
        

        # ну и трекаем обьекты

        # получаем списки:
        #   центроидов [ { object_id, (x, y) } ]
        #   метаданных [ { object_id, { ДИКТ ВСЕГО ЧТО ЕСТЬ В preds } } ]
        tracked_objects, tracked_meta = tracker.track(original_rgb_frame.copy(), detected_objects)

        # Вот тут можно зарисовывать все дела
        bboxed_frame = original_rgb_frame.copy()

        # Выводим красивые боксы со всей информацией
        for prediction in tracked_meta.values():
            
            # либо Имя либо Класс (надо только не айди а имя класса)
            label = fr"{prediction['class_name']} {float(prediction['confidence']):.03}"
            # {prediction['class_name']}

            bboxed_frame = plot_one_box(prediction['bbox'], bboxed_frame, label = label)

        tracked_frame = tracker.draw_trackable_objects_centroids(bboxed_frame, tracked_objects)
        total_frames += 1 
        
        #   На фрейме уже есть бокс и центроид

        #       meta
        # [{
        #     "class_name": self.names[int(result[5])],
        #     "bbox": [int(x) for x in result[:4].tolist()], #convert bbox results to int from float
        #     "confidence": float(result[4]),
        # }] 

        tracked_frame = cv2.cvtColor(tracked_frame, cv2.COLOR_BGR2RGB)
        writer.write(tracked_frame)

        for prediction in tracked_meta.values():
            last_prediction = prediction['class_name']
            confidence = prediction['confidence']
            res[last_prediction] += confidence
            bbox = prediction['bbox']

    cap.release()
    writer.release()

    value = 'Idk' if len(res) == 0 else max(res, key=res.get)
    return gr.update(value = '/tmp/tmp.mp4'), value

image_output = gr.Video(
    interactive = False,
    autoplay = True,
)

demo = gr.Interface(
    title="COSMOSTARS",
    description='<h2 style="text-align: center;">Сервис определения вида отходов</h2>',
    theme=gr.themes.Monochrome().set(body_background_fill='#7daae8'),
    fn=gradio_predict,
    inputs=gr.Video(
        label = 'Upload a video',
        sources = 'upload',
        ),
    outputs=[
        image_output,
        gr.Textbox(
            value = 'Example answer', 
            label="Answer"
        ),
        # output_file
    ],
    allow_flagging="never",
    live = True
)

app = gr.mount_gradio_app(app, demo, path=GRADIO_PATH)