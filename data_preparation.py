import cv2
import os
from glob import glob
from tqdm import tqdm
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def cut_videos(videos_path, start_second, end_second, save_path):
    videos = glob(videos_path)
    print('Всего видео:', len(videos))

    # режем видосы
    for i, video in enumerate(tqdm(videos)):
        class_name = video.split('/')[-2]
        video_name = video.split('/')[-1]
        if not os.path.exists(save_path + class_name):
            os.makedirs(save_path + class_name)

        try:
            ffmpeg_extract_subclip(
                video, 
                start_second, 
                end_second, 
                targetname=f'{save_path}{class_name}/{video_name}'
            )
        except:
            print(video)

    videos_cut = glob(f'{save_path}*/*')
    print('Всего урезанных видео:', len(videos_cut))


def get_fps(videos_path):
    fps_summ = 0
    videos = glob(videos_path)
    for video in videos:
        try:
            v = cv2.VideoCapture(video)
            fps = v.get(cv2.CAP_PROP_FPS)
            fps_summ += int(fps)
        except:
            pass

    print(fps_summ / len(videos))


def extract_frames(videos_path: str, save_path: str):
    videos = glob(videos_path)

    for video in tqdm(videos):
        class_name = video.split('/')[-2]
        video_name = video.split('/')[-1].split('.')[0]

        if not os.path.exists(save_path + class_name):
            os.makedirs(save_path + class_name)

        cap = cv2.VideoCapture(video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = 0
        num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_num % fps == 0:
                    cv2.imwrite(f'{save_path}{class_name}/{video_name}_{num}.jpg', frame)
                    num += 1
            else:
                break
            frame_num += 1
        
        cap.release()

def get_im_shapes(image_path):
    images = glob(image_path)
    for image in images[:10]:
        im = cv2.imread(image)
        print(im.shape)


def get_train_val(image_paths):
    class_folders = os.listdir(image_paths)
    for folder in class_folders:
        images = glob('data/train_frames/' + folder + '/*.jpg')
        images = sorted(images)

        train = images[:int(len(images)*0.9)]
        val = images[int(len(images)*0.9):]

        for image in train:
            if not os.path.exists('data/yolo_frames/train/' + folder):
                os.makedirs('data/yolo_frames/train/' + folder)
            shutil.copy(image, f"data/yolo_frames/train/{folder}")

        for image in val:
            if not os.path.exists('data/yolo_frames/val/' + folder):
                os.makedirs('data/yolo_frames/val/' + folder)
            shutil.copy(image, f"data/yolo_frames/val/{folder}")


def combine_images_with_txt():
    txts = glob('data/train_cars/obj_train_data_sec/*')
    images = glob('data/train_frames/Дерево/*')
    for txt in txts:
        for image in images:
            txt_name = txt.split('/')[-1].split('.')[0]
            image_name = image.split('/')[-1].split('.')[0]
            if txt_name == image_name:
                shutil.copy(image, 'data/train_cars/obj_train_data_sec')
                break


def get_train_val_detector():
    class_folder = glob('data/train_cars/*')

    for folder in tqdm(class_folder):
        images = glob(folder + '/*.jpg')
        images = sorted(images)

        train = images[:int(len(images)*0.9)]
        val = images[int(len(images)*0.9):]

        for image in train:
            labels = image.replace('.jpg', '.txt')
            shutil.copy(image, "data/yolo_detector/images/train")
            shutil.copy(labels, "data/yolo_detector/labels/train")

        for image in val:
            labels = image.replace('.jpg', '.txt')
            shutil.copy(image, "data/yolo_detector/images/val")
            shutil.copy(labels, "data/yolo_detector/labels/val")

    print('images train:', len(glob("data/yolo_detector/images/train/*")))
    print('labels train:', len(glob("data/yolo_detector/labels/train/*")))
    print('images val:', len(glob("data/yolo_detector/images/val/*")))
    print('labels val:', len(glob("data/yolo_detector/labels/val/*")))


        
if __name__ == "__main__":
    cut_videos(
        videos_path='data/train/*/*',
        start_second=130,
        end_second=140,
        save_path = 'data/train_cut/'
    )
    # get_fps(videos_path='data/train_cut/*/*')
    extract_frames(
        videos_path='data/train_cut/',
        save_path = 'data/train_frames/'
    )
    # get_im_shapes(image_path='data/train_frames/*/*')

    get_train_val(image_paths='data/train_frames')
    # combine_images_with_txt()
    get_train_val_detector()




