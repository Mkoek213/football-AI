import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

def track_players_distance_to_ball(model_path: str, source_video_path: str, target_video_path: str,
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
    try:
        # Load the YOLO model
        model = YOLO(model_path)

        # Create annotators for bounding boxes, traces, and labels
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        trace_annotator = sv.TraceAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=20, height=17
        )

        # Retrieve video information and initialize frame generator
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        frames_generator = sv.get_video_frames_generator(source_path=source_video_path)

        # Initialize a ByteTrack tracker for player tracking
        tracker = sv.ByteTrack()

        # Open the video sink for writing annotated frames
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in frames_generator:
                # Run YOLO model on the current frame
                result = model(frame, conf=confidence_threshold)[0]

                # Convert YOLO detections to a format usable by supervision
                detections = sv.Detections.from_ultralytics(result)

                # Update detections with the tracker
                detections = tracker.update_with_detections(detections)

                # Separate ball and player detections
                ball_detections = detections[detections.class_id == 0]
                player_detections = detections[detections.class_id == 2]
                player_detections = player_detections.with_nms(threshold=0.5, class_agnostic=True)

                # Annotate the frame with traces, bounding boxes, and labels
                annotated_frame = trace_annotator.annotate(
                    scene=frame.copy(),
                    detections=ball_detections
                )
                annotated_frame = ellipse_annotator.annotate(annotated_frame, player_detections)
                annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)

                # If ball detections are present
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

                # Write the annotated frame to the video sink
                sink.write_frame(frame=annotated_frame)

    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"File not found: {str(fnf_error)}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during video processing. Error: {str(e)}")


if __name__ == "__main__":
    """
    Main function to parse command-line arguments and run the distance calculation function.
    """
    # Set up argument parser to handle command-line input
    parser = argparse.ArgumentParser(description="Detect players and calculate their distance from the ball using a YOLO model.")
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file.')
    parser.add_argument('--source', type=str, required=True, help='Path to the source video file.')
    parser.add_argument('--target', type=str, default='result.mp4', help="Path to save the annotated video. Defaults to 'result.mp4'.")
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections. Default is 0.25.')
    parser.add_argument('--divider', type=float, default=24, help='Conversion factor from pixels to meters. For example for 1920x1080 video of 80m widht football field, 1 meter is 24 pixels')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the distance calculation process with the provided arguments
    track_players_distance_to_ball(
        model_path=args.model,
        source_video_path=args.source,
        target_video_path=args.target,
        confidence_threshold=args.conf,
        divider=args.divider
    )
