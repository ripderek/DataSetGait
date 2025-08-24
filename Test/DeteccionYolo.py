import cv2
from ultralytics import YOLO

#VIDEO_SOURCE = "Nuevos/ZamoraA/NoControlado/Lateral/1.mp4"
VIDEO_SOURCE = "Nuevos/YeranickM/Controlado/Frontal/1.mp4"
CONF_THRES = 0.35

WIN_NAME = "YOLO Tracking"
selected_id = None  # ID de la persona seleccionada
last_detections = []  # [(id, (x1,y1,x2,y2)), ...]

# Tamaño fijo deseado
FRAME_WIDTH = 1024
FRAME_HEIGHT = 600


def on_mouse(event, x, y, flags, userdata):
    global selected_id, last_detections
    if event == cv2.EVENT_LBUTTONDOWN:
        for tid, (x1, y1, x2, y2) in last_detections:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = tid
                print(f"[INFO] Persona seleccionada con ID {tid}")
                break

def main():
    global last_detections, selected_id

    model = YOLO("yolov8n.pt")  # usa YOLOv8 preentrenado en COCO
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el video.")
        return

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    print("[INFO] Click sobre una persona para seleccionarla. ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Detección y tracking integrado de YOLO
        results = model.track(frame, persist=True, classes=[0], conf=CONF_THRES, verbose=False)
        detections = []

        if results[0].boxes.id is not None:
            for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box[:4])
                tid = int(tid.item())
                detections.append((tid, (x1, y1, x2, y2)))

                # Dibujar todas las personas
                color = (0, 255, 0)
                label = f"ID {tid}"
                if tid == selected_id:
                    color = (0, 0, 255)  # seleccionado en rojo
                    label += " [SELECCIONADO]"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        last_detections = detections

        cv2.imshow(WIN_NAME, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
