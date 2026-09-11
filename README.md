# SFA-YOLO

## requirements

pip3 install pytorch, opencv-python, ultralytics

ultralytics is YOLOv13 version

## how to run:

### train:

edit the `train.py` file and train the model

`python train.py`

### val

edit `eval.py` to test the model trained

`python eval.py`

## others

### dataset

using SIMD dataset. check `dataset.yaml` for details.

### file location

* yaml location: `./yolov13.yaml`

* module location: add `./block/module.py` to `./ultralytics/nn/modules/block.py`


