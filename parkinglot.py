import cv2
import time
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv
from draw_line import draw_line


# Function to draw ROI
def draw_roi(event, x, y, flags, param):
    global roi, drawing, frame, roi_drawn

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = frame.copy()
            cv2.rectangle(img, roi[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi.append((x, y))
        roi_drawn = True
        cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

# Load YOLOv8 model
model = YOLO('yolov8l.pt')

# Initialize ROI
roi = []
drawing = False
roi_drawn = False

# Open video source
input_video = 'parking test video.mp4'
cap = cv2.VideoCapture(input_video)

# Dictionary to store dwell times
dwell_times = {}
start_times = {}
count_len = 0

# Read first frame and let user draw ROI
ret, frame = cap.read()

if not ret:
    print("Failed to read video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Resize frame
frame = cv2.resize(frame, (1020, 500))
cv2.imshow("Frame", frame)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_roi)
while not roi_drawn:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Main loop
def track_count_queue(input_video, output_video):
    # Set up video capture
    cap = cv2.VideoCapture(input_video)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Create a dictionary to keep track of objects that have crossed the line
    inside_box = {}

    # Get the original video resolution
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the resized resolution
    # resized_resolution = (1020, 500)

    # Open a video sink for the output video
    video_info = sv.VideoInfo.from_video_path(input_video)
    # video_info.resolution = resized_resolution  # Set the correct resolution
    with sv.VideoSink(output_video, video_info) as sink:

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame = cv2.resize(frame, (1020, 500))

                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id

                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                else:
                    track_ids = []

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and count objects crossing the line
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    x1, x2, y1, y2 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
                    # x1, x2, y1, y2 = x, x, y, y + h
                    cx, cy, = x + w/2, y + h/2

                    # box_width = 10
                    # box_height = h/10

                    # Check if the object is within the ROI
                    if (roi[0][0] < x1 < roi[1][0] and roi[0][1] < y1 < roi[1][1]) or (roi[0][0] < x2 < roi[1][0] and roi[0][1] < y2 < roi[1][1]):
                        if track_id not in inside_box:
                            inside_box[track_id] = True
                            start_times[track_id] = time.time()
                            dwell_times[track_id] = 0
                        else:
                            dwell_times[track_id] = time.time() - start_times[track_id]

                        # Annotate the object as it crosses the line
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # start_point = (int(x1-150), int(y+h/4))  # Top-right corner with some padding
                        # end_point = (int(x1), int((y+h)-h/2))  # Bottom-right corner of the box
                        # cv2.rectangle(annotated_frame, start_point, end_point, (255, 255, 255), cv2.FILLED)
                        # try:
                        #     cv2.putText(annotated_frame, f"ID: {track_id}, Dwell Time: {dwell_times[track_id]:.2f}s", (start_point, end_point), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                        # except:
                        #     print("Pass")
                        #     pass

                # Draw the ROI on the frame
                cv2.rectangle(annotated_frame, roi[0], roi[1], (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Total Slots: {14}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Empty Slots: {14-len(inside_box)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

               # Resize the annotated frame back to the original size before writing
                output_frame = cv2.resize(annotated_frame, (original_width, original_height))

                # Write the frame with annotations to the output video
                sink.write_frame(output_frame)
                cv2.imshow("frame", output_frame)

                if cv2.waitKey(10) & 0xFF == 27:
                    break
            else:
                break

    # Release the video capture
    cap.release()


output_video = 'output-parking test video.mp4'
track_count_queue(input_video, output_video)
cv2.destroyAllWindows()