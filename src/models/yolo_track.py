import cv2
from ultralytics import YOLO


def process_video(video_path, model_path, output_path):
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    # Loop through the video frames
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO tracking
        results = model.track(frame, persist=True,)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Tracking", annotated_frame)

        # Write the frame
        out.write(annotated_frame)

        # Display progress
        print(f"Processing frame {frame_count}/{total_frames}", end='\r')

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nVideo processing complete!")


if __name__ == "__main__":
    # Usage
    video_path = "datasets/videos/tactical_view.mov"
    # model_path = "yolo11_soccernet/train8/weights/best.pt"
    model_path = "yolov8x.pt"
    model_path = "yolo11_soccernet/pretrained_from_github/model1.pt"
    output_path = "outputs/tracking_2.mov"

    process_video(video_path, model_path, output_path)