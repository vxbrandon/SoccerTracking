import cv2
import numpy as np
from ultralytics import YOLO
# from tvcalib import Calib


class PlayerTrackingPipeline:
    def __init__(self, yolo_model_path, calibration_data_path):
        self.yolo_model = YOLO(yolo_model_path)
        # self.calibration = Calib.load(calibration_data_path)
        self.field_image = self.create_field_image()

    def detect_and_track_players(self, frame):
        results = self.yolo_model.track(frame, persist=True)
        return results[0]  # Assuming single image input

    def project_to_2d_field(self, tracked_objects):
        field_positions = []
        for obj in tracked_objects:
            if obj.id is None:  # Skip objects without tracking ID
                continue
            image_point = np.array(
                [obj.bbox[0] + obj.bbox[2] / 2, obj.bbox[1] + obj.bbox[3]])  # Bottom center of bounding box
            field_point = self.calibration.image_to_field(image_point)
            field_positions.append((int(obj.id), field_point))
        return field_positions

    def process_frame(self, frame):
        results = self.detect_and_track_players(frame)
        # field_positions = self.project_to_2d_field(results.boxes)
        field_positions = None
        return results, field_positions

    def create_field_image(self, width=800, height=500):
        field = np.zeros((height, width, 3), dtype=np.uint8)
        field[:, :] = (0, 128, 0)  # Green color for the field
        cv2.rectangle(field, (50, 50), (width - 50, height - 50), (255, 255, 255), 2)  # Field boundaries
        cv2.circle(field, (width // 2, height // 2), 50, (255, 255, 255), 2)  # Center circle
        return field

    def visualize(self, frame, results, field_positions):
        # Visualize on the original frame
        annotated_frame = results.plot()

        # Visualize on the 2D field
        field_vis = self.field_image.copy()
        for track_id, pos in field_positions:
            x, y = pos
            x = int(x * field_vis.shape[1])
            y = int(y * field_vis.shape[0])
            cv2.circle(field_vis, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(field_vis, f"ID: {track_id}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Combine the visualizations
        vis_frame = cv2.resize(annotated_frame, (
        field_vis.shape[1], int(annotated_frame.shape[0] * field_vis.shape[1] / annotated_frame.shape[1])))
        combined_vis = np.vstack((vis_frame, field_vis))
        return combined_vis

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (width, height + self.field_image.shape[0]))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results, field_positions = self.process_frame(frame)
            vis_frame = self.visualize(frame, results, field_positions)

            out.write(vis_frame)
            cv2.imshow('Player Tracking', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Usage example:
# pipeline = PlayerTrackingPipeline("path/to/yolo/model.pt", "path/to/calibration/data.json")
# pipeline.run("path/to/input_video.mp4", "path/to/output_video.mp4")