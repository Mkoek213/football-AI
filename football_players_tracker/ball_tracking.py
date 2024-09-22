import argparse
import cv2
from ultralytics import YOLO
from football_players_tracker.helper import SoccerPitchConfiguration, \
    ViewTransformer, draw_pitch, replace_outliers_based_on_distance, draw_paths_on_pitch
import supervision as sv
import numpy as np
from collections import deque
from tqdm import tqdm


def ball_tracking_view(model_path_pitch_detection: str, model_path_player_detection: str, source_video_path: str,
                       target_image_path: str, confidence_treshold: float):
    """
    Processes a football match video to detect the ball's position and plot the ball's trajectory on a soccer pitch image.

    Args:
    - model_path_pitch_detection (str): Path to the YOLO model for detecting the soccer pitch.
    - model_path_player_detection (str): Path to the YOLO model for detecting players and the ball.
    - source_video_path (str): Path to the input football match video.
    - target_image_path (str): Path to save the final image with the ball's trajectory plotted on the pitch.
    - confidence_treshold (float): Confidence threshold for the YOLO detection models.

    Returns:
    - None. Saves an annotated image to the specified target path.
    """
    try:
        # Initialize YOLO models for pitch and player/ball detection
        PITCH_DETECTION_MODEL = YOLO(model_path_pitch_detection)
        PLAYER_DETECTION_MODEL = YOLO(model_path_player_detection)

        # Configuration for drawing the soccer pitch
        CONFIG = SoccerPitchConfiguration()
        BALL_ID = 0
        MAXLEN = 5  # Maximum number of frames to smooth perspective transformation
        MAX_DISTANCE_THRESHOLD = 500  # Threshold to filter outliers in ball trajectory

        # Get video information (e.g., frame rate, resolution) and frame generator
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        frames_generator = sv.get_video_frames_generator(source_video_path)

        path_raw = []  # To store the raw ball coordinates over frames
        M = deque(maxlen=MAXLEN)  # Smoothing matrix for perspective transformation

        # Iterate through each frame of the video
        for frame in tqdm(frames_generator, total=video_info.total_frames):
            # Detect the ball in the current frame using the player/ball detection model
            result_ball = PLAYER_DETECTION_MODEL(frame, conf=confidence_treshold)[0]
            detections = sv.Detections.from_ultralytics(result_ball)

            # Extract ball detections and pad bounding boxes
            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # Detect the pitch in the current frame using the pitch detection model
            result_pitch = PITCH_DETECTION_MODEL(frame, conf=confidence_treshold)[0]
            key_points = sv.KeyPoints.from_ultralytics(result_pitch)

            # Filter key points based on confidence
            filter = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter]
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            # Initialize a view transformer to map video coordinates to pitch coordinates
            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )

            # Smooth the perspective transformation using an average of recent frames
            M.append(transformer.m)
            transformer.m = np.mean(np.array(M), axis=0)

            # Get ball's coordinates in the video frame and transform them to pitch coordinates
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            # Append the ball's pitch coordinates to the trajectory path
            path_raw.append(pitch_ball_xy)

        # Process the raw trajectory path and remove outliers
        path = [
            np.empty((0, 2), dtype=np.float32) if np.asarray(coordinates).shape[0] >= 2 else np.asarray(coordinates)
            for coordinates in path_raw
        ]
        path = [coordinates.flatten() for coordinates in path]
        path_res = replace_outliers_based_on_distance(path, MAX_DISTANCE_THRESHOLD)

        # Create an empty pitch image and draw the ball's trajectory on it
        annotated_frame = draw_pitch(CONFIG)
        annotated_frame = draw_paths_on_pitch(
            config=CONFIG,
            paths=[path_res],
            color=sv.Color.WHITE,
            pitch=annotated_frame
        )

        # Save the annotated image (with the ball's trajectory) to the target path
        cv2.imwrite(target_image_path, annotated_frame)

    except Exception as e:
        # Error handling: print the exception message and raise the exception for debugging
        print(f"Error in ball tracking view creation: {e}")
        raise


if __name__ == "__main__":
    try:
        # Set up argument parser to handle command-line input
        parser = argparse.ArgumentParser(description="Create ball tracking view from a video using a YOLO model.")
        parser.add_argument('--pitch_detection_model', type=str, required=True,
                            help='Path to the YOLO pitch detection model file.')
        parser.add_argument('--player_detection_model', type=str, required=True,
                            help='Path to the YOLO player detection model file.')
        parser.add_argument('--source', type=str, required=True, help='Path to the source video file.')
        parser.add_argument('--target', type=str, default='result.jpg',
                            help="Path to save the ball tracking image. Defaults to 'result.jpg'.")
        parser.add_argument('--conf', type=float, default=0.25,
                            help='Confidence threshold for detections. Default is 0.25')

        # Parse command-line arguments
        args = parser.parse_args()

        # Run the ball tracking view function with the provided arguments
        ball_tracking_view(
            model_path_pitch_detection=args.pitch_detection_model,
            model_path_player_detection=args.player_detection_model,
            source_video_path=args.source,
            target_image_path=args.target,
            confidence_treshold=args.conf
        )

    except Exception as exp:
        # Error handling: Print any exceptions that occur during script execution
        print(f"Error: {str(exp)}")
