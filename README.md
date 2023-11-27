# yolo-heatmaps
A utility for generating explanatory heatmaps from YOLOv8 (https://github.com/ultralytics/ultralytics) object detection results 
using Layerwise Relevance Propagation (LRP/CRP).

```
yolo = YOLO('yolov8x.pt')
detection = yolo(image)

...

lrp = YOLOv8LRP(yolo, power=2, eps=1e-05, device='cuda')

explanation_lrp_person = lrp.explain(image, cls='person', contrastive=False)
explanation_lrp_cat = lrp.explain(image, cls='cat', contrastive=False)
```

## LRP Heatmaps

![image](https://github.com/akarasman/yolo-heatmaps/assets/56434833/db92eacd-b6d2-4b6f-86a2-cc3fe3d8fad8)

## CRP Heatmaps

![image](https://github.com/akarasman/yolo-heatmaps/assets/56434833/140e8e6a-e589-450f-8c09-6b05f94fbeeb)
