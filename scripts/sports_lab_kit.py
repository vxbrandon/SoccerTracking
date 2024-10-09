import sportslabkit as slk

from sportslabkit.mot import SORTTracker

# Initialize your camera and models
cam = slk.Camera(path_to_mp4)
det_model = slk.detection_model.load('YOLOv8x', imgsz=640)
motion_model = slk.motion_model.load('KalmanFilter', dt=1/30, process_noise=10000, measurement_noise=10)

# Configure and execute the tracker
tracker = SORTTracker(detection_model=det_model, motion_model=motion_model)
tracker.track(cam[:100])
res = tracker.to_bbdf()

save_path = "assets/tracking_results.mp4"
res.visualize_frames(cam.video_path, save_path)

# The tracking data is now ready for analysis