import numpy as np
import pandas as pd
import os
import json
import anyio
import logging
from pathlib import Path
from evidently.metrics import DataDriftTable
from evidently.report import Report
from torchvision import transforms
from PIL import Image
from segmentationsuim.train import NUM_CLASSES
from segmentationsuim.data import download_dataset, get_dataloaders
from segmentationsuim.api import extract_features, extract_pred_features
from google.cloud import storage
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

BUCKET_NAME = "mlops_g71_monitoring"

logger = logging.getLogger(__name__)


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("data_drift.html")


def lifespan(app: FastAPI):
    global training_data

    cfg = {
        "image_transformations": {"image_size": 256},
        "data_loader": {"batch_size": 16, "workers": 4, "split_ratio": 0.8},
    }

    logging.basicConfig(level=logging.INFO)
    logging.info("Downloading dataset...")
    download_dataset()
    data_path = "data/raw"

    img_size = cfg["image_transformations"]["image_size"]
    image_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)])

    train_loader, _, _ = get_dataloaders(
        data_path=data_path,
        use_processed=False,
        image_transform=image_transform,
        mask_transform=mask_transform,
        batch_size=cfg["data_loader"]["batch_size"],
        num_workers=cfg["data_loader"]["workers"],
        split_ratio=cfg["data_loader"]["split_ratio"],
    )

    train_features = []
    for img, pred in train_loader:
        img = img.numpy().squeeze()
        pred = pred.numpy().squeeze()
        features = extract_features(img)
        pred_features = extract_pred_features(pred)
        train_features.append(np.concatenate([features, pred_features]))

    feature_columns = ["Average Brightness", "Contrast", "Sharpness"]
    for i in range(NUM_CLASSES):
        feature_columns.append(f"Class {i} Proportion")

    reference_data_df = np.array(train_features)

    # Creating column names for the features
    training_data = pd.DataFrame(reference_data_df, columns=feature_columns)

    # Ensuring features in reference_df are numeric
    training_data[feature_columns] = training_data[feature_columns].astype(float)

    yield

    del training_data


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n)

    # Get all prediction files in the directory
    files = directory.glob("results/predictions/prediction_*.json")

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed
    all_data = []
    for file in latest_files:
        with file.open() as f:
            data = json.load(f)
            all_data.append(data)
    df = pd.DataFrame(all_data)

    # Drop the "Time" column if it exists
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    return df.astype(float)


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""
    bucket = storage.Client().bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix="predictions/prediction_")
    blobs = sorted(blobs, key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    os.makedirs("results/predictions", exist_ok=True)

    for blob in latest_blobs:
        destination_path = os.path.join("results", blob.name)
        blob.download_to_filename(destination_path)


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5) -> None:
    prediction_data = load_latest_files(Path("."), n=n)
    run_analysis(training_data, prediction_data)

    async with await anyio.open_file("data_drift.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
