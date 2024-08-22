from ultralytics import YOLO
import torch
def train_model(model_path, data_path, save_path, freeze_layers, epochs=100, imgsz=640, batch_size=12, run_dir='runs'):

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


    freeze = [f"model.{x}." for x in range(freeze_layers+1)]

    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False

    for k, v in model.named_parameters():
        print(k, v.requires_grad)

    # Define training parameters
    train_params = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        'plots': True,
        'patience': 100,
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
        model_path='models/yolov5s.pt',  # Path to the pre-trained model
        data_path='datasets/data.yaml',  # Path to the data configuration file for training
        save_path='models/yolov5s_transfer_based_model.pt',  # Path to save the trained model
        freeze_layers = 10,
        epochs=1,  # Number of training epochs for training
        run_dir='runs/fine_tuning_yolov5s_freezed_based_model'  # Specify the directory for this training run
    )