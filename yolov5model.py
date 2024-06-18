import torch

def load_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
    return model