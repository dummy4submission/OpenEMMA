from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import cv2


model = YOLO("weights/yolo11x.pt")
image_path = "/home/cyqian/YOLO3D/eval/nuscenes/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151696662404.jpg"
results = model(image_path)

img = cv2.imread(image_path)
for r in results:
    
    annotator = Annotator(img)
    
    boxes = r.boxes
    for box in boxes:
        
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
        
img = annotator.result()  
cv2.imwrite('YOLO V11x Detection.png', img)     