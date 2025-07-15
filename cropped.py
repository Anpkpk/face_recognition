import cv2
import cvzone
import time
import supervision as sv
from ultralytics import YOLO


facemodel = YOLO(r'C:\VSCode\Python\face_recognition\yolov8m-face.pt')

def crop_face(image):
    img = cv2.imread(image)
    results = facemodel.predict(img)

    for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1
                cvzone.cornerRect(img, [x1, y1, w, h], l=9, rt=3)
                cropped = img[y1:y2, x1:x2]
    
    return cropped

face = crop_face(r'C:\VSCode\Python\face_recognition\test.jpg')
sv.plot_image(face)