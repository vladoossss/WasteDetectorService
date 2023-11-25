from typing import Iterable, Optional
import cv2
import base64
import numpy.typing as npt


def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)'''
    return [
                [
                    {
                    "class": int(pred[5]),
                    "class_name": model.model.names[int(pred[5])],
                    "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
                    "confidence": float(pred[4]),
                    }
                for pred in result
                ]
            for result in results.xyxy
            ]

def plot_one_box(bbox: Iterable[float], im: npt.NDArray, color: Iterable[int] = (0, 0, 200), label: Optional[str] = None, line_thickness: int = 3) -> npt.NDArray:

    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    
    # tl = line_thickness or round(
    #     0.002 * (im.shape[0] + im.shape[1]) / 2
    # ) + 1  # line/font thickness
    tl = line_thickness 
    
    top_left, bottom_right = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(im, top_left, bottom_right, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if not label:
        return im
    
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    bottom_right = top_left[0] + t_size[0], top_left[1] - t_size[1] - 3
    # cv2.rectangle(im, top_left, bottom_right, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(im, label, (top_left[0], top_left[1] - 2), cv2.FONT_HERSHEY_COMPLEX, tl / 3, [0, 0, 200], 
                thickness=tf)

    return im

def draw_poly(cvimage, poly):
    for segment in poly: # in case there are more than one segmrnt
        vertices = segment.reshape((-1, 1, 2))
        cv2.polylines(cvimage, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

def base64EncodeImage(img):
    ''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

    return im_b64