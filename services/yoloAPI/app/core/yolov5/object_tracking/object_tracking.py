from typing import Any, Dict, List, Optional, OrderedDict, Tuple
from app.core.yolov5.object_tracking.centroidtracker import CentroidTracker
from app.core.yolov5.object_tracking.trackableobject import TrackableObject
import cv2
from collections import defaultdict
import dlib

import numpy.typing as npt

class ObjectTracker:
    """ Трекинг обьектов на кадрах видео """

    def __init__(self, skip_frames: int = 0, min_tracked_frames_for_drawing: int = 0, max_disappeared_frames = 5, max_tracked_distance = 600):
        self.skip_frames = skip_frames
        self.total_frames = 0
        self.trackers = []
        self.trackable_objects = {}
        self.rects = []
        self.names = []

        self.object_counter = defaultdict(int)
        self.centroid_tracker = CentroidTracker(
            max_disappeared_frames = max_disappeared_frames, 
            max_tracked_distance = max_tracked_distance
        )
        self.min_tracked_frames_for_drawing = min_tracked_frames_for_drawing

    def __draw_object(self, frame: npt.NDArray, centroid: Tuple[int, int], objectID: int) -> npt.NDArray:
        """ Зарисовка ID объекта и его центроида на кадре """
        frame = frame.copy()

        # frame = cv2.putText(
        #     frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2
        # )
        frame = cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 200), -1)
        
        return frame
    
    def __create_trackers(self, rgb_frame: npt.NDArray, new_object_dicts) -> None:
        """ Создает обьекты центроид-трекеры """
        self.trackers = []
        # цикл по детектам
        for obj in new_object_dicts:
            bbox = obj['bbox']
            (startX, startY, endX, endY) = bbox

            # Создаем dlib коробку-обьект из координат полученных выше
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb_frame, rect)

            # добавляем трекер к списку трекеров
            self.trackers.append((tracker, obj))

    def __update_trackers(self, rgb_frame: npt.NDArray) -> None:
        """ Обновляет положение обьектов центроид-трекеров """
        for tracker, obj in self.trackers:
            # Получаем обновленную позицию
            tracker.update(rgb_frame)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # добавляем координаты в список описывающих коробок
            self.rects.append((startX, startY, endX, endY, obj))

    def __bind_centroid_trackers_to_objects(self) -> OrderedDict:
        """ Связывает описывающие коробки с центроид-трекерами """
        # Использованеи центроид трекера для связки старых обьектов-центроидов
        # 	(чтобы не создавать новые) и новых границ обьектов
        objects, meta = self.centroid_tracker.update(self.rects)
        for objectID, centroid in objects.items():
            
            # TODO: а зачем нам вообще trackable_objects?...

            to = self.trackable_objects.get(objectID, None)
            self.object_counter[objectID] += 1
            
            # Если отсуствует - создаем
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                to.centroids.append(centroid)

            self.trackable_objects[objectID] = to

        return objects, meta

    def draw_trackable_objects_centroids(self, frame: npt.NDArray, objects: OrderedDict) -> npt.NDArray:
        """ Зарисовывает центроид-трекеры на кадре """
        frame = frame.copy()

        for (objectID, centroid) in objects.items():
            if self.object_counter[objectID] > self.min_tracked_frames_for_drawing:
                frame = self.__draw_object(frame, centroid, objectID)
        
        return frame
    
    def get_last_trackable_objects(self) -> OrderedDict:
        """ Возвращает список последних трекаемых обьектов """
        return self.objects, self.names

    def track(
        self, 
        rgb_frame: Optional[npt.NDArray], 
        new_object_dicts: List[Dict[str, Any]], 
    ) -> Tuple[OrderedDict, OrderedDict]:
        """ Главный метод трекинга обьектов """
        self.rects.clear()

        # bboxes = [obj['bbox'] for obj in new_object_dicts]
        # names = [obj['person_name'] for obj in new_object_dicts]

        #   Пайплайн работает на балансировки между детекцией и трекингом
        #       детекция выполняется раз в skip_frames кадров, в остальное время только трекинг
        if len(new_object_dicts) != 0:
            self.__create_trackers(rgb_frame, new_object_dicts)
        # else:
            # self.__update_trackers(rgb_frame)
        self.__update_trackers(rgb_frame)

        self.objects, self.meta = self.__bind_centroid_trackers_to_objects()

        self.total_frames += 1

        return self.objects, self.meta
