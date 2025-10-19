from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("yolov8n.pt")
person_class_id = 0
confidence_threshold = 0.4
iou_threshold = 0.7
sel=0

def process_frame_results(results_generator):
    if not results_generator:
        return None
        
    try:
        result = next(iter(results_generator))
    except StopIteration:
        return None
    
    if result is None or result.orig_img is None:
        return None
            
    frame = result.orig_img
    
    person_id_counter = 1
    
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = str(person_id_counter)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2
        text_y = y1 - 10
        padding = 3 
        
        cv2.rectangle(frame, (text_x - padding, text_y - text_size[1] - padding), 
                      (text_x + text_size[0] + padding, text_y + padding), 
                      color, -1)
        
        cv2.putText(frame, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        person_id_counter += 1

    return frame

if(sel==0):
    video_path = r"C:\Users\pooji\OneDrive\Desktop\croud count\project\milestone2\6387-191695740.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file at: {video_path}")

    results_generator = model.predict(
        source=video_path,
        classes=[person_class_id], 
        conf=confidence_threshold,
        iou=iou_threshold,
        stream=True,
        verbose=False,
        tracker='bytetrack.yaml'
    )
    
    for result in results_generator:
        processed_frame = process_frame_results([result])
        if processed_frame is not None:
            cv2.imshow("Crowd Counter", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release() 
    cv2.destroyAllWindows()
    
else:
    cam_cap = cv2.VideoCapture(0)
    if not cam_cap.isOpened():
        print("cant capture")
        exit()
        
    while True:
        ret, frame = cam_cap.read()
        if not ret:
            print("failed to capture frame")
            break
            
        results = model.predict(
            source=frame,
            classes=[person_class_id], 
            conf=confidence_threshold,
            iou=iou_threshold,
            stream=False,
            verbose=False,
            tracker='bytetrack.yaml'
        )
        
        processed_frame = process_frame_results(results)
        
        if processed_frame is not None:
            cv2.imshow("Crowd Counter", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_cap.release()
    cv2.destroyAllWindows()