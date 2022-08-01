# YOLOv5 Heatmaps

![front](https://user-images.githubusercontent.com/56434833/162103265-7f9401b8-251b-4a82-aead-5eb8893cfe9e.png)

A utility for generating heatmaps of YOLOv5 https://github.com/ultralytics/yolov5 using Layerwise Relevance Propagation (LRP/CRP).
Pytorch implementation based on: https://github.com/moboehle/Pytorch-LRP

## Install

```bash
git clone https://github.com/akarasman/yolo-heatmaps/
cd yolo-heatmaps
pip install -r requirements.txt
```

## CLI Use Example

```bash
python3 explain.py --source=data/images/so-and-so.jpg --weights=yolov5s.pt --explain-class='person'
```

Run results saved to runs/explain/exp(# of run)

## Arguments

```bash
  # explain.py is built on detect.py module from YOLOv5, lrp options are :
  
  --power POWER         Power exponent applied to weights and inputs
  --contrastive         Use contrastive relevance (CRP)
  --b1 B1               Visualization parameter for CRP - multiplier of primal part
  --b2 B2               Visualization parameter for CRP - multiplier of dual part
  --explain-class EXPLAIN_CLASS
                        Class to explain
  --conf                Confidence threshold on object
  --max-class-only      Max class only
  --box-xywh BOX_XYWH [BOX_XYWH ...]
                        Box to restrict investigation (X,Y,W,H format)
  --smooth-ks SMOOTH_KS
                        Box to restrict investigation (X,Y,W,H format)
  --box-xyxy BOX_XYXY [BOX_XYXY ...]
                        Box to restrict investigation (X,Y,X,Y format)
  --cmap CMAP           Explanation color map (default set to seismic/magma when contrastive / non-contrastive
```

Current version only supports YOLOv5s-x models.


Please cite our paper if you plan on using code from this repository for your work 

```
@inproceedings{inproceedings,
author = {Karasmanoglou, Apostolos and Antonakakis, Marios and Zervakis, Michalis},
year = {2022},
month = {06},
pages = {1-6},
title = {Heatmap-based Explanation of YOLOv5 Object Detection with Layer-wise Relevance Propagation},
doi = {10.1109/IST55454.2022.9827744}
}
```
