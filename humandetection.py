import cv2
import numpy as np
from ultralytics import YOLO


def video_capture(video_path):
    source = cv2.VideoCapture(video_path)
    model = YOLO("yolo11n.pt")

    win_name = "CCTV Analysis"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    catalogued_ids = set()

    while True:
        has_frame, frame = source.read()
        if not has_frame:
            break

        results = model.track(frame, persist=True, verbose=False, classes=[0], tracker="bytetrack.yaml")
        
        id_frame_counts = {}

        if results[0].boxes.id is not None:

            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.int().cpu().tolist()

            for track_id, box in zip(track_ids, boxes):
                
                id_frame_counts[track_id] = id_frame_counts.get(track_id, 0) + 1

                if id_frame_counts[track_id] and track_id not in catalogued_ids:
                    catalogued_ids.add(track_id)

                    print(f"New person detected! ID: {track_id}")

                    #could save crop of face here
                
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Total Catalogued: {len(catalogued_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_capture("MallCCTV.mp4")