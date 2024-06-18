import cv2
import torch
from PIL import Image
import numpy as np

from yolov5model import load_model

from typing import List
import argparse

def pre_process(video_path: str, model=None, save_frames=False) -> List[np.ndarray]:
    '''
    Pre-process the video frames to be fed into the model
    :param video_path: str: name of the video file
    :param model: torch model: model to be used for prediction
    :return: list: list of frames
    '''

    # load model if not provided
    model = load_model() if model is None else model

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 256
    height = 256

    # to do raise the error for video name
    # handle video name error

    if save_frames:
        outpath = video_path.split('.')[0] + '_processed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
    
    processed_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform inference
        results = model(pil_img)

        # Extract bounding boxes and class labels
        bboxes = results.xyxy[0].numpy()
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            if int(cls) == 0:  # Class 0 is person for YOLOv5
                # Crop the frame to the bounding box
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Resize the cropped frame to the original frame size
                resized_frame = cv2.resize(cropped_frame, (width, height))
                
                # Write the frame with bounding box to the output video
                if save_frames: out.write(resized_frame)
                processed_frames.append(resized_frame)
                break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument(
        '-v',
        '--video_path', 
        type=str, 
        required=True, 
        help='Path to the video file'
    )

    arg.add_argument(
        '-s',
        '--save_frames',
        type=bool,
        default=True,
        help='Save the processed frames to a video file'
    )

    args = vars(arg.parse_args())
    pre_process(**args)