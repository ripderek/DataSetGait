#la deteccion con Yolo nano es mas rapida y efectiva pero tiene mayor costo computacional
from ultralytics import YOLO
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

# Cargar YOLO versión liviana (n = nano)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("Nuevos/VelezV/Controlado/Frontal/1.mp4")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (config.ANCHO, config.ALTO))

    results = model(frame)
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # Clase 0 = persona en COCO
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imshow("YOLOv8n Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
