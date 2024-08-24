import string
import argparse
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import os


def detect_players(model_path: str, source_video_path: str, target_video_path: str, confidence_treshold: float):
    """
    Detect players in a video using a YOLO model and annotate the video with bounding boxes and labels.

    Parameters:
    - model_path (str): The file path to the YOLO model.
    - source_video_path (str): The file path to the source video to be processed.
    - target_video_path (str): The file path where the annotated video will be saved.
    - confidence_treshold (float): The confidence threshold for the YOLO model to filter detections.
    """

    # Check if the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check if the source video file exists
    if not os.path.isfile(source_video_path):
        raise FileNotFoundError(f"Source video file not found: {source_video_path}")

    # Initialize the YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model from {model_path}. Error: {str(e)}")

    # Initialize the annotators for bounding boxes and labels
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),  # Set colors for bounding boxes
        thickness=2  # Set thickness of bounding box borders
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
        # Set background colors for labels
        text_color=sv.Color.from_hex("#000000")  # Set text color for labels
    )

    # Retrieve video information such as frame count, resolution, etc.
    try:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve video information from {source_video_path}. Error: {str(e)}")

    # Initialize the video sink for writing the annotated video frames
    try:
        video_sink = sv.VideoSink(target_video_path, video_info=video_info)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize video sink for {target_video_path}. Error: {str(e)}")

    # Generate frames from the source video and process each frame
    try:
        frame_generator = sv.get_video_frames_generator(source_video_path)
        with video_sink:  # Ensure that the video sink is properly closed after processing
            for frame in tqdm(frame_generator, total=video_info.total_frames):  # Progress bar for frame processing
                # Perform object detection on the current frame
                result = model(frame, conf=confidence_treshold)[0]
                detections = sv.Detections.from_ultralytics(result)

                # Prepare labels for the detected objects
                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(detections["class_name"], detections.confidence)
                ]

                # Create a copy of the frame to annotate
                annotated_frame = frame.copy()
                # Annotate the frame with bounding boxes
                annotated_frame = box_annotator.annotate(annotated_frame, detections)
                # Annotate the frame with labels
                annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
                # Write the annotated frame to the output video
                video_sink.write_frame(annotated_frame)
    except Exception as exp:
        raise RuntimeError(f"An error occurred during video processing. Error: {str(exp)}")


if __name__ == "__main__":
    """
    Main function to parse command-line arguments and run the detect_players function.
    """

    # Set up argument parser to handle command-line input
    parser = argparse.ArgumentParser(description="Detect players in a video using a YOLO model.")
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file.')
    parser.add_argument('--source', type=str, required=True, help='Path to the source video file.')
    parser.add_argument('--target', type=str, default='result.mp4',
                        help='Path to save the annotated video. Defaults to "result.mp4".')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the detection process with the provided arguments
    try:
        detect_players(
            model_path=args.model,
            source_video_path=args.source,
            target_video_path=args.target,
            confidence_treshold=args.conf
        )
    except Exception as exp:
        print(f"Error: {str(exp)}")
