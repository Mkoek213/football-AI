import argparse
import cv2
from ultralytics import YOLO
from football_players_tracker.helper import SoccerPitchConfiguration, extract_crops, TeamClassifier, \
    resolve_goalkeppers_team_id, ViewTransformer, draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
import supervision as sv
import numpy as np


def voronoi_radar_view(model_path_pitch_detection: str, model_path_player_detection: str, source_video_path: str,
               target_video_path: str, confidence_treshold: float):
    """
    Processes a football match video to generate a Voronoi radar visualization.
    It detects the pitch, players, and ball, then overlays a semi-transparent radar view
    showing player/team positions using Voronoi diagrams.

    Args:
    - model_path_pitch_detection (str): Path to the YOLO model for pitch detection.
    - model_path_player_detection (str): Path to the YOLO model for player and ball detection.
    - source_video_path (str): Path to the input football match video.
    - target_video_path (str): Path to save the output video with radar overlay.
    - confidence_treshold (float): Confidence threshold for the detection models.
    """
    try:
        # Load YOLO models for pitch and player/ball detection
        PITCH_DETECTION_MODEL = YOLO(model_path_pitch_detection)
        PLAYER_DETECTION_MODEL = YOLO(model_path_player_detection)

        # Set up the soccer pitch configuration
        CONFIG = SoccerPitchConfiguration()

        # Define class IDs for detection targets
        BALL_ID = 0
        GOALKEPPER_ID = 1
        PLAYER_ID = 2
        REFEREE_ID = 3

        # Extract player crops for team classification training
        crops = extract_crops("models/yolov8x_transfer_based_model.pt", source_video_path=source_video_path)
        team_classifier = TeamClassifier()
        team_classifier.fit(crops)

        # Initialize tracker for tracking multiple objects (players, ball, referees)
        tracker = sv.ByteTrack()

        # Get the video info and frame generator for processing
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        frames_generator = sv.get_video_frames_generator(source_path=source_video_path)
        tracker.reset()

        # Set up radar dimensions and transparency
        frame_width, frame_height = video_info.width, video_info.height
        pitch_height = 200  # Height of the radar overlay
        pitch_width = frame_width // 2  # Radar width (half of the video width)
        pitch_x_offset = (frame_width - pitch_width) // 2  # Horizontal offset to center the radar
        alpha = 0.5  # Transparency of the radar overlay

        # Frame counter for limiting frames (e.g., for testing)
        frame_counter = 0
        max_frames = 100  # Process only 10 frames for testing

        # Open video output sink to write annotated frames
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in frames_generator:
                # Stop after max_frames have been processed
                if frame_counter >= max_frames:
                    break

                # Detect players and the ball in the frame using the player detection model
                result = PLAYER_DETECTION_MODEL(frame, conf=confidence_treshold)[0]
                detections = sv.Detections.from_ultralytics(result)

                # Isolate and pad ball detections for better visibility
                ball_detections = detections[detections.class_id == BALL_ID]
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                # Filter out ball detections and apply Non-Max Suppression (NMS) to remaining detections
                all_detections = detections[detections.class_id != BALL_ID]
                all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                all_detections = tracker.update_with_detections(all_detections)

                # Separate detections by role (players, goalkeepers, referees)
                players_detections = all_detections[all_detections.class_id == PLAYER_ID]
                goalkeepers_detections = all_detections[all_detections.class_id == GOALKEPPER_ID]
                referee_detections = all_detections[all_detections.class_id == REFEREE_ID]

                # Crop player images and classify teams based on uniforms
                players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
                players_detections.class_id = team_classifier.predict(players_crops)

                # Resolve goalkeeper team association based on proximity to players
                goalkeepers_detections.class_id = resolve_goalkeppers_team_id(
                    players_detections, goalkeepers_detections
                )

                # Detect pitch key points to map the video frame to the pitch layout
                result_pitch = PITCH_DETECTION_MODEL(frame, conf=confidence_treshold)[0]
                key_points = sv.KeyPoints.from_ultralytics(result_pitch)
                filter = key_points.confidence[0] > 0.5
                frame_reference_points = key_points.xy[0][filter]
                pitch_reference_points = np.array(CONFIG.vertices)[filter]

                # Transform player/ball coordinates from the frame to the radar pitch layout
                view_transformer = ViewTransformer(
                    source=frame_reference_points,
                    target=pitch_reference_points
                )

                # Get ball and player positions in pitch coordinates
                frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_ball_xy = view_transformer.transform_points(frame_ball_xy)

                frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_players_xy = view_transformer.transform_points(frame_players_xy)

                frame_referees_xy = referee_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_referees_xy = view_transformer.transform_points(frame_referees_xy)

                # Draw the soccer pitch and Voronoi radar overlay (team positions visualized as Voronoi cells)
                pitch = draw_pitch(config=CONFIG)
                pitch = draw_pitch_voronoi_diagram(
                    config=CONFIG,
                    team_1_xy=pitch_players_xy[players_detections.class_id == 0],
                    team_2_xy=pitch_players_xy[players_detections.class_id == 1],
                    team_1_color=sv.Color.from_hex('00BFFF'),
                    team_2_color=sv.Color.from_hex('FF1493'),
                    pitch=pitch
                )

                # Resize the radar pitch and overlay it at the bottom of the video frame
                pitch_resized = cv2.resize(pitch, (pitch_width, pitch_height))
                match_and_pitch_combined = frame.copy()
                roi = match_and_pitch_combined[frame_height - pitch_height: frame_height,
                      pitch_x_offset: pitch_x_offset + pitch_width]
                cv2.addWeighted(pitch_resized, alpha, roi, 1 - alpha, 0, roi)

                # Write the annotated frame to the output video
                sink.write_frame(frame=match_and_pitch_combined)

                # Increment the frame counter
                frame_counter += 1

    except Exception as e:
        # Catch and print errors for debugging
        print(f"Error during radar view creation: {e}")
        raise


if __name__ == "__main__":
    try:
        # Parse command-line arguments to provide paths for models, video files, and other parameters
        parser = argparse.ArgumentParser(description="Create radar from a video using a YOLO model.")
        parser.add_argument('--pitch_detection_model', type=str, required=True,
                            help='Path to the YOLO pitch detection model file.')
        parser.add_argument('--player_detection_model', type=str, required=True,
                            help='Path to the YOLO player detection model file.')
        parser.add_argument('--source', type=str, required=True, help='Path to the source video file.')
        parser.add_argument('--target', type=str, default='result.mp4',
                            help="Path to save the annotated video. Defaults to 'result.mp4'.")
        parser.add_argument('--conf', type=float, default=0.25,
                            help='Confidence threshold for detections. Default is 0.25')

        # Execute the main function using parsed arguments
        args = parser.parse_args()
        voronoi_radar_view(
            model_path_pitch_detection=args.pitch_detection_model,
            model_path_player_detection=args.player_detection_model,
            source_video_path=args.source,
            target_video_path=args.target,
            confidence_treshold=args.conf
        )

    except Exception as exp:
        # Print error messages if something goes wrong during execution
        print(f"Error: {str(exp)}")
