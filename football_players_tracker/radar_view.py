import argparse
import cv2
from ultralytics import YOLO
from football_players_tracker.helper import SoccerPitchConfiguration, extract_crops, TeamClassifier, \
    resolve_goalkeppers_team_id, ViewTransformer, draw_pitch, draw_points_on_pitch
import supervision as sv
import numpy as np


def radar_view(model_path_pitch_detection: str, model_path_player_detection: str, source_video_path: str,
               target_video_path: str,
               confidence_treshold: float):
    """
    This function processes a football match video to overlay a radar (visualized pitch with player and ball positions)
    on the match screen. The radar is semi-transparent and is placed at the bottom of the match video.

    Args:
    - model_path_pitch_detection (str): Path to the YOLO model for detecting the pitch.
    - model_path_player_detection (str): Path to the YOLO model for detecting players and the ball.
    - source_video_path (str): Path to the input video of the football match.
    - target_video_path (str): Path to save the output video with the radar.
    - confidence_treshold (float): Confidence threshold for detection models.
    """
    try:
        # Initialize YOLO models for pitch and player/ball detection
        PITCH_DETECTION_MODEL = YOLO(model_path_pitch_detection)
        PLAYER_DETECTION_MODEL = YOLO(model_path_player_detection)

        # Initialize soccer pitch configuration
        CONFIG = SoccerPitchConfiguration()

        # Object class IDs in detection model
        BALL_ID = 0
        GOALKEPPER_ID = 1
        PLAYER_ID = 2
        REFEREE_ID = 3

        # Extract player crops and train team classifier based on those crops
        crops = extract_crops("models/yolov8x_transfer_based_model.pt", source_video_path=source_video_path)
        team_classifier = TeamClassifier()
        team_classifier.fit(crops)

        # Initialize tracker for players and objects
        tracker = sv.ByteTrack()

        # Get video information (dimensions, frame rate, etc.)
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        frames_generator = sv.get_video_frames_generator(source_path=source_video_path)
        tracker.reset()

        # Set pitch radar dimensions and transparency
        frame_width, frame_height = video_info.width, video_info.height
        pitch_height = 200  # Height of the radar
        pitch_width = frame_width // 2  # Width of the radar (half of the video width)
        pitch_x_offset = (frame_width - pitch_width) // 2  # Center the radar horizontally
        alpha = 0.5  # Transparency level for overlaying radar

        # Frame counter for limiting the number of processed frames (for testing purposes)
        frame_counter = 0
        max_frames = 10  # Save only first 10 frames for testing

        # Open a video writer for the output video
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in frames_generator:
                # Break loop if the frame limit is reached
                if frame_counter >= max_frames:
                    break

                # Run player and ball detection on the frame
                result = PLAYER_DETECTION_MODEL(frame, conf=confidence_treshold)[0]
                detections = sv.Detections.from_ultralytics(result)

                # Filter and pad ball detections
                ball_detections = detections[detections.class_id == BALL_ID]
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                # Filter non-ball detections and run Non-Max Suppression (NMS)
                all_detections = detections[detections.class_id != BALL_ID]
                all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                all_detections = tracker.update_with_detections(all_detections)

                # Separate player, goalkeeper, and referee detections
                players_detections = all_detections[all_detections.class_id == PLAYER_ID]
                goalkeepers_detections = all_detections[all_detections.class_id == GOALKEPPER_ID]
                referee_detections = all_detections[all_detections.class_id == REFEREE_ID]

                # Classify player team based on their crops
                players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
                players_detections.class_id = team_classifier.predict(players_crops)

                # Resolve goalkeeper's team ID
                goalkeepers_detections.class_id = resolve_goalkeppers_team_id(
                    players_detections, goalkeepers_detections
                )

                # Detect pitch key points for perspective transformation
                result_pitch = PITCH_DETECTION_MODEL(frame, conf=confidence_treshold)[0]
                key_points = sv.KeyPoints.from_ultralytics(result_pitch)
                filter = key_points.confidence[0] > 0.5
                frame_reference_points = key_points.xy[0][filter]
                pitch_reference_points = np.array(CONFIG.vertices)[filter]

                # Transform coordinates from video frame to pitch
                view_transformer = ViewTransformer(
                    source=frame_reference_points,
                    target=pitch_reference_points
                )

                # Get ball and players' positions in pitch coordinates
                frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_ball_xy = view_transformer.transform_points(frame_ball_xy)
                frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_players_xy = view_transformer.transform_points(frame_players_xy)
                frame_referees_xy = referee_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_referees_xy = view_transformer.transform_points(frame_referees_xy)

                # Draw pitch and radar points
                pitch = draw_pitch(config=CONFIG)
                pitch = draw_points_on_pitch(config=CONFIG, xy=pitch_ball_xy, face_color=sv.Color.WHITE,
                                             edge_color=sv.Color.BLACK, radius=10, pitch=pitch)
                pitch = draw_points_on_pitch(config=CONFIG, xy=pitch_players_xy[players_detections.class_id == 0],
                                             face_color=sv.Color.from_hex("00BFFF"), edge_color=sv.Color.BLACK,
                                             radius=10, pitch=pitch)
                pitch = draw_points_on_pitch(config=CONFIG, xy=pitch_players_xy[players_detections.class_id == 1],
                                             face_color=sv.Color.from_hex("FF1493"), edge_color=sv.Color.BLACK,
                                             radius=10, pitch=pitch)
                pitch = draw_points_on_pitch(config=CONFIG, xy=pitch_referees_xy,
                                             face_color=sv.Color.from_hex("FFD700"), edge_color=sv.Color.BLACK,
                                             radius=10, pitch=pitch)

                # Resize the pitch radar and overlay on the bottom of the match frame
                pitch_resized = cv2.resize(pitch, (pitch_width, pitch_height))
                match_and_pitch_combined = frame.copy()
                roi = match_and_pitch_combined[frame_height - pitch_height: frame_height,
                      pitch_x_offset: pitch_x_offset + pitch_width]
                cv2.addWeighted(pitch_resized, alpha, roi, 1 - alpha, 0, roi)

                # Save the frame to video
                sink.write_frame(frame=match_and_pitch_combined)

                # Increment the frame counter
                frame_counter += 1

    except Exception as e:
        print(f"Error during radar view creation: {e}")
        raise


if __name__ == "__main__":
    try:
        # Set up argument parser to handle command-line input
        parser = argparse.ArgumentParser(description="Create radar from a video using a YOLO model.")
        parser.add_argument('--pitch_detection_model', type=str, required=True,
                            help='Path to the YOLO pitch detection model file.')
        parser.add_argument('--player_detection_model', type=str, required=True,
                            help='Path to the YOLO player detection model file.')
        parser.add_argument('--source', type=str, required=True, help='Path to the source video file.')
        parser.add_argument('--target', type=str, default='result.mp4',
                            help="Path to save the annotated video. Defaults to 'result.mp4.")
        parser.add_argument('--conf', type=float, default=0.25,
                            help='Confidence threshold for detections. Default is 0.25')

        # Parse command-line arguments
        args = parser.parse_args()

        # Run the radar_view function with the provided arguments
        radar_view(
            model_path_pitch_detection=args.pitch_detection_model,
            model_path_player_detection=args.player_detection_model,
            source_video_path=args.source,
            target_video_path=args.target,
            confidence_treshold=args.conf
        )

    except Exception as exp:
        # Print any exceptions that occur during execution
        print(f"Error: {str(exp)}")
