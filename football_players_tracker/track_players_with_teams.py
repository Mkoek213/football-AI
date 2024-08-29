import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import helper


def track_players_with_teams(model_path: str, source_video_path: str, target_video_path: str,
                             confidence_threshold: float, divider: float):
    """
    Detect players in a video, calculate the distance between each player and a ball,
    and annotate the results in the video using a YOLO model.

    Parameters:
    model_path (str): Path to the YOLO model file.
    source_video_path (str): Path to the input video file.
    target_video_path (str): Path to save the annotated video.
    confidence_threshold (float): Confidence threshold for detections.
    divider (float): Conversion factor from pixels to meters for distance calculation.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Extract crops (images) of players from the video for team classification
    crops = helper.extract_crops(model=model_path, source_video_path=source_video_path)
    team_classifier = helper.TeamClassifier()
    team_classifier.fit(crops)

    # Create annotators for bounding boxes, traces, and labels
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    trace_annotator = sv.TraceAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1493']),
        thickness=2
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER
    )

    # Retrieve video information and initialize frame generator
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_path=source_video_path)

    # Initialize a ByteTrack tracker for player tracking
    tracker = sv.ByteTrack()
    tracker.reset()

    # Open the video sink for writing annotated frames
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in frames_generator:
            # Run YOLO model on the current frame to detect objects
            result = model(frame, conf=confidence_threshold)[0]

            # Convert YOLO detections to a format usable by the Supervision library
            detections = sv.Detections.from_ultralytics(result)

            # Update detections using the tracker to maintain object identities across frames
            detections = tracker.update_with_detections(detections)

            # Separate ball detections (class_id = 0) from player detections (class_id = 2)
            ball_detections = detections[detections.class_id == 0]

            # Filter out ball detections and perform non-maximum suppression (NMS) on remaining detections
            all_detections = detections[detections.class_id != 0]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)

            # Extract player detections and predict their team using the team classifier
            player_detections = all_detections[all_detections.class_id == 2]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
            player_detections.class_id = team_classifier.predict(players_crops).astype(int)

            # Handle goalkeeper detections and assign them to teams using a helper function
            goalkepper_detections = all_detections[all_detections.class_id == 1]
            goalkepper_detections.class_id = helper.resolve_goalkeppers_team_id(player_detections,
                                                                                goalkepper_detections).astype(int)

            # Adjust referee detections by subtracting 1 from their class_id
            referee_detections = all_detections[all_detections.class_id == 3]
            referee_detections.class_id = (referee_detections.class_id - 1).astype(int)

            # Merge all detections into a single set for annotation
            all_detections = sv.Detections.merge([player_detections, goalkepper_detections, referee_detections])

            # Generate labels for the tracked objects
            labels = [
                f"{tracker_id}"
                for tracker_id
                in all_detections.tracker_id
            ]

            # Annotate the frame with traces, bounding boxes, and labels
            annotated_frame = trace_annotator.annotate(
                scene=frame.copy(),
                detections=ball_detections
            )
            annotated_frame = ellipse_annotator.annotate(annotated_frame, all_detections)
            annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, all_detections, labels=labels)

            # If ball detections are present, calculate distances to each player and annotate them
            if len(ball_detections) > 0:
                # Select the first detected ball (assuming there's only one ball)
                ball_bbox = ball_detections.xyxy[0]
                ball_center = np.array([
                    (ball_bbox[0] + ball_bbox[2]) / 2,
                    (ball_bbox[1] + ball_bbox[3]) / 2
                ])

                # Calculate distances from the ball to each player
                for player_bbox in player_detections.xyxy:
                    player_center = np.array([
                        (player_bbox[0] + player_bbox[2]) / 2,
                        (player_bbox[1] + player_bbox[3]) / 2
                    ])
                    distance = np.linalg.norm(ball_center - player_center)
                    distance_meters = distance / divider

                    # Annotate the distance on the frame
                    text = f"Distance: {distance_meters:.2f} meters"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (int(player_center[0]), int(player_center[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1
                    )

            # Write the annotated frame to the output video
            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":
    """
    Main function to parse command-line arguments and run the distance calculation function.
    """
    # Set up argument parser to handle command-line input
    parser = argparse.ArgumentParser(
        description="Detect players and calculate their distance from the ball using a YOLO model.")
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file.')
    parser.add_argument('--source', type=str, required=True, help='Path to the source video file.')
    parser.add_argument('--target', type=str, default='result.mp4',
                        help="Path to save the annotated video. Defaults to 'result.mp4'.")
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections. Default is 0.25.')
    parser.add_argument('--divider', type=float, default=24,
                        help='Conversion factor from pixels to meters. For example for 1920x1080 video of 80m wide football field, 1 meter is 24 pixels')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the distance calculation process with the provided arguments
    track_players_with_teams(
        model_path=args.model,
        source_video_path=args.source,
        target_video_path=args.target,
        confidence_threshold=args.conf,
        divider=args.divider
    )
