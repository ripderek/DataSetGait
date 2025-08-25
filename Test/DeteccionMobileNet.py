import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
import cv2

# Cargar modelo preentrenado (descargable de OpenCV Zoo o GitHub)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

cap = cv2.VideoCapture("Nuevos/VelezV/Controlado/Frontal/1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Redimensionar el frame antes del blob (ej. 800x600 o 640x480)
    frame = cv2.resize(frame, (config.ANCHO, config.ALTO))  
    (h, w) = frame.shape[:2]

    # Crear el blob a partir del frame reducido
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # ID 15 = persona en COCO
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("MobileNet SSD Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
