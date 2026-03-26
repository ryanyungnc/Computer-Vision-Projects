import cv2
import os
import sys
import threading
from dotenv import load_dotenv
from google.cloud import vision

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

results = []
analyzing = False

LIKELIHOOD_LABELS = {
    0: "UNKNOWN",
    1: "VERY_UNLIKELY",
    2: "UNLIKELY",
    3: "POSSIBLE",
    4: "LIKELY",
    5: "VERY_LIKELY",
}


def analyze_frame(frame):
    global results, analyzing
    try:
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        _, buffer = cv2.imencode(".jpg", frame)
        image = vision.Image(content=buffer.tobytes())
        response = client.face_detection(image=image)
        results = list(response.face_annotations)
    except Exception as e:
        print(f"Analysis error: {e}")
    finally:
        analyzing = False


def draw_results(frame, scale=1.0):
    for face in results:
        vertices = face.bounding_poly.vertices
        x = int(vertices[0].x * scale)
        y = int(vertices[0].y * scale)
        x2 = int(vertices[2].x * scale)
        y2 = int(vertices[2].y * scale)

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        emotions = {
            "joy": face.joy_likelihood,
            "sorrow": face.sorrow_likelihood,
            "anger": face.anger_likelihood,
            "surprise": face.surprise_likelihood,
        }
        dominant = max(emotions, key=lambda k: emotions[k])
        likelihood = LIKELIHOOD_LABELS.get(emotions[dominant], "UNKNOWN")
        label = f"emotion: {dominant} ({likelihood})"

        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return frame


def analyze_cctv_video(video_path):
    global results, analyzing
    source = cv2.VideoCapture(video_path)
    if not source.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    win_name = "CCTV Analysis"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    counter = 0

    while cv2.waitKey(1) != 27:  # ESC
        has_frame, frame = source.read()
        if not has_frame:
            break

        if counter % 10 == 0 and not analyzing:
            analyzing = True
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            thread = threading.Thread(target=analyze_frame, args=(small_frame.copy(),))
            thread.daemon = True
            thread.start()

        try:
            frame = draw_results(frame, scale=4.0)
        except Exception as e:
            print(f"Draw error: {e}")

        counter += 1
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)


def webcamCapture():
    global results, analyzing
    s = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        s = int(sys.argv[1])

    source = cv2.VideoCapture(s)
    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    counter = 0

    while cv2.waitKey(1) != 27:  # ESC
        has_frame, frame = source.read()
        if not has_frame:
            break

        flipped_frame = cv2.flip(frame, 1)

        if counter % 5 == 0 and not analyzing:
            analyzing = True
            small_frame = cv2.resize(flipped_frame, (0, 0), fx=0.25, fy=0.25)
            thread = threading.Thread(target=analyze_frame, args=(small_frame.copy(),))
            thread.daemon = True
            thread.start()

        try:
            flipped_frame = draw_results(flipped_frame, scale=4.0)
        except Exception as e:
            print(f"Draw error: {e}")

        counter += 1
        cv2.imshow(win_name, flipped_frame)

    source.release()
    cv2.destroyWindow(win_name)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        analyze_cctv_video(sys.argv[1])
    else:
        webcamCapture()
