import cv2
import torch
def train():
    from ultralytics import YOLO
    model = YOLO("esc.yaml")
    args = {
        'data': './homo.yaml',
        'epochs': 300,
        'device': '0',
        'workers': 4,
        'name': 'escnet',
        'save_period': 10,
        'cache' : 'ram',
        'val': True,
        'seed': 23,
        'amp' : True,
        'deterministic' : True
    }
    results = model.train(**args)
if __name__ == '__main__':
    train()
