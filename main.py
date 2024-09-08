import cv2
import numpy as np
from sort import Sort
from ultralytics import YOLO
from pytube import YouTube
import os

def download_youtube_video(url, output_path="downloads"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').first()
    video_path = stream.download(output_path=output_path)
    print(f"Downloaded video to {video_path}")
    
    return video_path

def main(youtube_url):
    video_path = download_youtube_video(youtube_url)
    model = YOLO('yolov8n.pt').to('cpu')  # YOLOv8n is a lightweight version

    tracker = Sort()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        detections = []
        for result in results:
            for detection in result.boxes:
                xyxy = detection.xyxy[0].cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
                conf = detection.conf[0].cpu().numpy()  # Confidence score
                cls = detection.cls[0].cpu().numpy()    # Class ID (0 for person)

                if int(cls) == 0:
                    detections.append([*xyxy, conf])

        tracked_objects = tracker.update(np.array(detections))

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Tracking', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    youtube_url = 'https://www.youtube.com/watch?v=V9YDDpo9LWg'  # Replace with a YouTube video link
    main(youtube_url)
