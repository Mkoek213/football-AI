import torch
import numpy as np
import supervision as sv
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import umap.umap_ as umap
from sklearn.cluster import KMeans
from typing import Generator, Iterable, List, TypeVar

V = TypeVar("V")


def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    """
    Splits an iterable sequence into smaller batches of a specified size.

    Parameters:
    sequence (Iterable[V]): The sequence to split into batches.
    batch_size (int): The number of elements per batch.

    Yields:
    Generator[List[V], None, None]: A generator yielding batches as lists of elements.
    """
    batch_size = max(batch_size, 1)  # Ensure batch_size is at least 1
    current_batch = []

    # Iterate through each element in the sequence
    for element in sequence:
        # Yield the batch when it reaches the specified size
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)

    # Yield any remaining elements in the last batch
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier to distinguish between different teams based on image crops using a vision model.
    """

    def __init__(self, batch_size: int = 32):
        """
        Initializes the TeamClassifier with a specified batch size and loads the required models.

        Parameters:
        batch_size (int): The number of images to process in a batch. Default is 32.
        """
        # Select device based on availability of GPU
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        # Load the pre-trained vision model and processor
        self.features_model = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224').to(self.device)
        self.processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')

        # Initialize UMAP for dimensionality reduction and KMeans for clustering
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extracts feature embeddings from image crops using the vision model.

        Parameters:
        crops (List[np.ndarray]): A list of image crops as NumPy arrays.

        Returns:
        np.ndarray: A NumPy array of feature embeddings.
        """
        # Convert the crops from OpenCV format to Pillow format
        crops = [sv.cv2_to_pillow(crop) for crop in crops]

        # Create batches of crops to process
        batches = create_batches(crops, self.batch_size)
        data = []

        # Disable gradient computation for inference
        with torch.no_grad():
            for batch in tqdm(batches):
                # Process the batch through the model
                inputs = self.processor(images=batch, return_tensors='pt').to(self.device)
                outputs = self.features_model(**inputs)

                # Compute mean embeddings for the batch
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        # Concatenate the embeddings of all batches into a single array
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fits the classifier to the provided image crops by extracting features, reducing dimensions,
        and clustering the data.

        Parameters:
        crops (List[np.ndarray]): A list of image crops as NumPy arrays.
        """
        # Extract features from the crops
        data = self.extract_features(crops)

        # Reduce dimensions using UMAP
        projections = self.reducer.fit_transform(data)

        # Fit the KMeans clustering model to the reduced data
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predicts the team classification for the provided image crops.

        Parameters:
        crops (List[np.ndarray]): A list of image crops as NumPy arrays.

        Returns:
        np.ndarray: An array of predicted team labels.
        """
        if len(crops) == 0:
            return np.array([])

        # Extract features from the crops
        data = self.extract_features(crops)

        # Reduce dimensions using UMAP
        projections = self.reducer.transform(data)

        # Predict team labels using the trained clustering model
        return self.cluster_model.predict(projections)
