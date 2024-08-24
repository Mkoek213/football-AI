from ultralytics import YOLO
import torch
def train_model(model_path, data_path, save_path, time, freeze_layers, epochs=100, imgsz=1280, batch_size=1, run_dir='runs'):

    """
    Train or fine-tune a YOLO model.

    Parameters:
    - model_path (str): Path to the model file.
    - data_path (str): Path to the data configuration file.
    - save_path (str): Path to save the trained model.
    - epochs (int): Number of training epochs. Default is 100.
    - imgsz (int): Image size for training. Default is 640.
    - batch_size (int): Batch size for training. Default is 16.
    """

    # Load the YOLO model
    model = YOLO(model_path)

    # Define training parameters
    train_params = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        'plots': True,
        'time': time,
        'freeze': freeze_layers,
        'patience': 5,
        'cache': True,
        'project': run_dir  # Specify the directory for storing runs
    }

    print("Starting training...")
    results = model.train(**train_params)
    # Save the pre-trained model
    model.save(save_path)

    # Print training results
    print(results)


if __name__ == "__main__":
    train_model(
        model_path='../models/yolov8x.pt',  # Path to the pre-trained model
        data_path='datasets/data.yaml',  # Path to the data configuration file for training
        save_path='models/yolov8x_transfer_based_model.pt',  # Path to save the trained model
        freeze_layers = 10,
        time = 2,
        epochs=50,  # Number of training epochs for training
        run_dir='runs/fine_tuning_yolov8x_freezed_based_model'  # Specify the directory for this training run
    )