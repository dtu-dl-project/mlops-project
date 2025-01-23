from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from typing import Literal
from torchvision import transforms
from hydra import initialize, compose
from omegaconf import DictConfig
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import uuid
import numpy as np
import json
import logging
from segmentationsuim.train import NUM_CLASSES
from datetime import datetime
from collections.abc import Generator
import time
from google.cloud import storage
from segmentationsuim.train import UNetModule, unet, Trans

BUCKET_NAME = "mlops_g71_monitoring"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prometheus metrics
api_request_counter = Counter("api_requests_total", "Total number of API requests received")
api_error_counter = Counter("api_errors_total", "Total number of API errors encountered")
classification_time_histogram = Histogram("classification_time_seconds", "Time taken to classify an image")

# Directories for uploads and results
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

logger = logging.getLogger(__name__)


def lifespan(app: FastAPI) -> Generator[None]:
    logging.basicConfig(level=logging.INFO)

    global unet_model, transformer_model, unet_image_size, transformer_image_size
    logging.info("Loading unet model...")
    # Load the unet model
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="unet.yaml")
    unet_image_size = cfg.image_transformations.image_size
    try:
        unet_model = load_model(cfg)
    except ValueError as e:
        # Increment the error counter
        api_error_counter.inc()
        raise HTTPException(status_code=400, detail=str(e))

    logging.info("Loading transformer model...")
    # Load the transformer model
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="transformer.yaml")
    transformer_image_size = cfg.image_transformations.image_size
    try:
        transformer_model = load_model(cfg)
    except ValueError as e:
        # Increment the error counter
        api_error_counter.inc()
        raise HTTPException(status_code=400, detail=str(e))

    with open("prediction_database.csv", "w") as file:
        file.write("Time,Average Brightness,Contrast,Sharpness")
        for i in range(NUM_CLASSES):
            file.write(f",Class {i} Proportion")
        file.write("\n")

    yield


app = FastAPI(lifespan=lifespan)


def download_model_from_gcp(filename: str):
    """
    Download the model checkpoint from Google Cloud Storage.
    """

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{filename}")
    blob.download_to_filename(os.path.join("models", filename))
    logging.info(f"Model {filename} downloaded from GCP bucket {BUCKET_NAME}.")


def load_model(cfg: DictConfig):
    """
    Load the appropriate model based on the configuration file.
    """
    checkpoint_path = os.path.join(cfg.checkpoints.dirpath, cfg.checkpoints.filename)

    if cfg.checkpoints.filename not in os.listdir(cfg.checkpoints.dirpath):
        logging.info("Downloading model from GCP")
        download_model_from_gcp(cfg.checkpoints.filename)

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    if cfg.training.model == "unet":
        model = UNetModule.load_from_checkpoint(checkpoint_path, unet=unet).to(DEVICE)
    elif cfg.training.model == "transformer":
        model = Trans.load_from_checkpoint(
            checkpoint_path  # Image size from the config
        ).to(DEVICE)
    else:
        raise ValueError("Invalid model type in configuration. Use 'unet' or 'transformer'.")
    model.eval()
    return model


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the API for segment your underwater image. You can choose between two different models: unet and transformer. Upload your image and see the magic!!"
    }


@app.get("/metrics")
async def metrics():
    """Expose metrics for Prometheus scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def extract_features(image):
    """Extract basic image features from a single image."""
    avg_brightness = np.mean(image)
    contrast = np.std(image)
    sharpness = np.mean(np.abs(np.gradient(image)))
    return avg_brightness, contrast, sharpness


def extract_pred_features(pred):
    """Extract basic prediction features from a prediction."""
    pred_features = []
    total_pixels = pred.size
    for i in range(NUM_CLASSES):
        pred_features.append(np.sum(pred == i) / total_pixels)
    return np.array(pred_features)


def save_prediction_to_gcp(
    time: str,
    avg_brightness: float,
    contrast: float,
    sharpness: float,
    pred_features: np.ndarray,
):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    data = {
        "Time": time,
        "Average Brightness": str(avg_brightness),
        "Contrast": str(contrast),
        "Sharpness": str(sharpness),
    }
    for i in range(NUM_CLASSES):
        data[f"Class {i} Proportion"] = str(pred_features[i])
    blob = bucket.blob(f"predictions/prediction_{time}.json")
    blob.upload_from_string(json.dumps(data), content_type="application/json")
    logging.info(f"Prediction data for {time} saved to GCP bucket {BUCKET_NAME}.")


@app.post("/predict/")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: Literal["unet", "transformer"] = Query(..., description="Choose the model: unet or transformer"),
):
    """
    API endpoint for predicting with UNet or Transformer models.
    - Upload an image.
    - Choose the model type (`unet` or `transformer`).
    """
    # Increment the API request counter
    api_request_counter.inc()

    # Start timing for the histogram
    start_time = time.time()

    # Save the uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Preprocess the uploaded image
    image = Image.open(input_path).convert("RGB")

    if model_type == "unet":
        img_size = unet_image_size
        model = unet_model
    elif model_type == "transformer":
        img_size = transformer_image_size
        model = transformer_model

    image_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    # Predict using the selected model
    with torch.no_grad():
        if model_type == "unet":
            output = model.unet(img_tensor)
            output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
            prediction = output.argmax(dim=1).squeeze().cpu().numpy()
        elif model_type == "transformer":
            img_proc = model.processor(images=img_tensor, return_tensors="pt").pixel_values.to(DEVICE)
            logits = model.model(img_proc).logits
            prediction = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Save the prediction as an image
    result_path = os.path.join(RESULT_FOLDER, f"{uuid.uuid4()}_prediction.png")
    plt.imsave(result_path, prediction, cmap="tab20")

    # Save the prediction to the database
    now = str(datetime.now())
    features = extract_features(np.array(img_tensor.cpu().squeeze()))
    pred_features = extract_pred_features(prediction)
    background_tasks.add_task(save_prediction_to_gcp, now, *features, pred_features)

    del model

    # Record classification time in the histogram
    logging.info(f"Classification time: {time.time() - start_time}")
    classification_time_histogram.observe(time.time() - start_time)

    # Return the prediction image
    return FileResponse(result_path, media_type="image/png", filename=f"prediction_{model_type}.png")
