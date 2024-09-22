# Football AI Project

This project focuses on tracking football players and analyzing various aspects of their movement and positioning using AI-powered models. The project utilizes YOLO-based models to detect players, track ball movements, and generate visualizations like Voronoi graphs, radar views, and player-ball distance tracking. The results include annotated videos and images of football matches with ball and player tracking visualizations.

## Project Structure

- **`football_players_tracker/`**: Contains the main scripts for running different tracking and analysis functions.
- **`helper.py`**: Contains helper functions required by the main scripts, such as soccer pitch configurations, drawing utilities, and transformation functions.
- **`result_videos/`**: Stores all the generated result videos and images, including player detection, ball tracking, and other visualizations.
- **`notebooks/`**: Contains `testing.ipynb`, which you can use to test new functions interactively.

---

## Main Scripts and Usage

Below is a description of each script inside the `football_players_tracker` directory and how to run them from the terminal.

### 1. `ball_tracking.py`
![ball_tracking_view_result](https://github.com/user-attachments/assets/f9b945de-1a92-46c3-9c20-3ae226637988)


**Description**: Processes a football match video to detect the ball's position and plot the ball's trajectory on a soccer pitch image.

**Command to run**:
```bash
python radar_view.py --pitch_detection_model "models/yolov8x_pitch_detection.pt" --player_detection_model "models/yolov8x_transfer_based_model.pt"  --source "test_videos/121364_0.mp4" --target "result_videos/radar_view_result.mp4" --conf 0.25
```

### 2. `detect_players.py`
![players_detection_screen](https://github.com/user-attachments/assets/14f1099d-17a5-475e-a255-ecb9d45afbf8)


**Description**: Detect players in a video using a YOLO model and annotate the video with bounding boxes and labels.

**Command to run**:
```bash
python detect_players.py --model "models/yolov8x_transfer_based_model.pt" --source "test_videos/121364_0.mp4" --target "result_videos/players_detection_result.mp4" --conf 0.3
```

### 3. `detect_players_with_trace.py`
![players_detection_with_trace_screen](https://github.com/user-attachments/assets/0f7fbe23-06d3-4779-87c8-e757bb93d5d4)


**Description**: Detect players in a video and annotate their positions with trace lines using a YOLO model.

**Command to run**:
```bash
python detect_players_with_trace.py --model "models/yolov8x_transfer_based_model.pt" --source "test_videos/121364_0.mp4" --target "result_videos/players_tracking_result.mp4" --conf 0.25
```

### 4. `radar_view.py`
![radar_view_screen](https://github.com/user-attachments/assets/36eb0ad4-8e9a-45ea-ac71-f683371acd83)


**Description**:     This function processes a football match video to overlay a radar (visualized pitch with player and ball positions)
    on the match screen. The radar is semi-transparent and is placed at the bottom of the match video.

**Command to run**:
```bash
python radar_view.py --pitch_detection_model "models/yolov8x_pitch_detection.pt" --player_detection_model "models/yolov8x_transfer_based_model.pt"  --source "test_videos/121364_0.mp4" --target "result_videos/radar_view_result.mp4" --conf 0.25
```

### 5. `track_player_distance_to_ball.py`
![players_tracking_with_ball_distance_screen](https://github.com/user-attachments/assets/cdbc6644-9f84-4174-9175-2145adcb1b70)


**Description**:         Detect players in a video, calculate the distance between each player and a ball,
    and annotate the results in the video using a YOLO model.

**Command to run**:
```bash
python track_players_distance_to_ball.py --model "models/yolov8x_transfer_based_model.pt" --source "test_videos/121364_0.mp4" --target "result_videos/players_tracking_with_ball_distance_result.mp4" --conf 0.25 --divider 24
```

### 6. `track_players_with_teams.py`
![players_tracking_with_teams_screen](https://github.com/user-attachments/assets/137ce4b8-0d85-4a7d-b44f-e26c90b43b58)


**Description**:     Detect players in a video, calculate the distance between each player and a ball,
    and annotate the results in the video using a YOLO model.

**Command to run**:
```bash
python track_players_with_teams.py --model "models/yolov8x_transfer_based_model.pt" --source "test_videos/121364_0.mp4" --target "result_videos/players_tracking_with_teams_result.mp4" --conf 0.25 --divider 24
```

### 7. `voronoi_graph_view.py`
![voronoi_graph_view_screen](https://github.com/user-attachments/assets/241af688-4012-457b-9cfd-6bb5243b3f2d)


**Description**:        Processes a football match video to generate a Voronoi radar visualization.
    It detects the pitch, players, and ball, then overlays a semi-transparent radar view
    showing player/team positions using Voronoi diagrams.

**Command to run**:
```bash
python voronoi_graph_view.py --pitch_detection_model "models/yolov8x_pitch_detection.pt" --player_detection_model "models/yolov8x_transfer_based_model.pt"  --source "test_videos/121364_0.mp4" --target "result_videos/voronoi_graph_view_result.mp4" --conf 0.25
```



## Additional Information

- **Helper Functions**: The `helper.py` file contains all the utility functions used by the main scripts, such as pitch configurations, player path drawing, and transformation utilities.

- **Result Videos and Images**: All the output videos and images generated by the scripts will be saved in the `result_videos/` directory.

- **Testing New Functions**: You can find a `testing.ipynb` notebook inside the `notebooks/` directory, where you can test new functions and interact with the data before integrating it into the main project.


## Installation

To run the scripts, first install the required dependencies:

```bash
poetry install
```

## Inspiration
https://github.com/SkalskiP
