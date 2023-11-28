# yolo-heatmaps
A utility for generating explanatory heatmaps from YOLOv8 (https://github.com/ultralytics/ultralytics) object detection results 
using Layerwise Relevance Propagation (LRP/CRP) (https://iphome.hhi.de/samek/pdf/MonXAI19.pdf).

```
yolo = YOLO('yolov8x.pt') 
detection = yolo(image) # Image is a C x H x W processed tensor

...

lrp = YOLOv8LRP(yolo, power=2, eps=1e-05, device='cuda')

# Explanation is a C x H x W tensor
explanation_lrp_person = lrp.explain(image, cls='person', contrastive=False)
explanation_lrp_cat = lrp.explain(image, cls='cat', contrastive=False)
```

## LRP Heatmaps

![image](https://github.com/akarasman/yolo-heatmaps/assets/56434833/db92eacd-b6d2-4b6f-86a2-cc3fe3d8fad8)

## CRP Heatmaps

![image](https://github.com/akarasman/yolo-heatmaps/assets/56434833/140e8e6a-e589-450f-8c09-6b05f94fbeeb)

If you are planning to utilize this repo in your research kindly cite the following work:

```
@INPROCEEDINGS{9827744,
  author={Karasmanoglou, Apostolos and Antonakakis, Marios and Zervakis, Michalis},
  booktitle={2022 IEEE International Conference on Imaging Systems and Techniques (IST)}, 
  title={Heatmap-based Explanation of YOLOv5 Object Detection with Layer-wise Relevance Propagation}, 
  year={2022},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/IST55454.2022.9827744}
}
```
