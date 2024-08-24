import argparse
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import os


def detect_players_with_trace(model_path: str, source_video_path: str, target_video_path: str,
                              confidence_treshold: float):
    """
    Detect players in a video and annotate their positions with trace lines using a YOLO model.

    Parameters:
    model_path (str): Path to the YOLO model file.
    source_video_path (str): Path to the input video file.
    target_video_path (str): Path to save the annotated video.
    confidence_treshold (float): Confidence threshold for detections.
    """

    try:
        # Load the YOLO model
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load the YOLO model. Error: {str(e)}")

    try:
        # Create annotators for bounding boxes, labels, and traces
        box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )

        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex("#000000")
        )

        trace_annotator = sv.TraceAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
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
                result = model(frame, conf=confidence_treshold)[0]

                # Convert YOLO detections to a format usable by supervision
                detections = sv.Detections.from_ultralytics(result)

                # Update detections with the tracker
                detections = tracker.update_with_detections(detections)

                # Create labels with class names and confidence scores
                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(detections["class_name"], detections.confidence)
                ]

                # Annotate the frame with traces, bounding boxes, and labels
                annotated_frame = trace_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections
                )
                annotated_frame = box_annotator.annotate(annotated_frame, detections)
                annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

                # Write the annotated frame to the video sink
                sink.write_frame(frame=annotated_frame)

    except FileNotFoundError as fnf_error:
        # Handle file not found errors
        raise FileNotFoundError(f"File not found: {str(fnf_error)}")
    except Exception as e:
        # Handle any other exceptions during processing
        raise RuntimeError(f"An error occurred during video processing. Error: {str(e)}")


if __name__ == "__main__":
    """
    Main function to parse command-line arguments and run the detect_players_with_trace function.
    """

    try:
        # Set up argument parser to handle command-line input
        parser = argparse.ArgumentParser(description="Detect players in a video using a YOLO model.")
        parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file.')
        parser.add_argument('--source', type=str, required=True, help='Path to the source video file.')
        parser.add_argument('--target', type=str, default='result.mp4',
                            help="Path to save the annotated video. Defaults to 'result.mp4.")
        parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections. '
                                                                     'Default is 0.25')

        # Parse the command-line arguments
        args = parser.parse_args()

        # Run the detection process with the provided arguments
        detect_players_with_trace(
            model_path=args.model,
            source_video_path=args.source,
            target_video_path=args.target,
            confidence_treshold=args.conf
        )

    except Exception as exp:
        # Print any exceptions that occur during execution
        print(f"Error: {str(exp)}")
