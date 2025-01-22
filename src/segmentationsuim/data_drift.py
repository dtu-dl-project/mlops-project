import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.report import Report
from torchvision import transforms
from PIL import Image
from segmentationsuim.train import NUM_CLASSES
from segmentationsuim.data import download_dataset, get_dataloaders
from segmentationsuim.api import extract_features, extract_pred_features

cfg = {
    "image_transformations": {"image_size": 256},
    "data_loader": {"batch_size": 16, "workers": 4, "split_ratio": 0.8},
}

transform = transforms.Compose([transforms.ToTensor()])

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

current_data_csv = pd.read_csv("prediction_database.csv")
current_data = current_data_csv.drop(columns=["Time"])

reference_data_df = np.column_stack((train_features, ["reference_data"] * len(train_features)))

# Creating column names for the features
reference_df = pd.DataFrame(reference_data_df, columns=feature_columns + ["Dataset"])

# Ensuring features in reference_df are numeric
reference_df[feature_columns] = reference_df[feature_columns].astype(float)

# Adding the "Dataset" column to the current data
current_df = current_data.copy()
current_df["Dataset"] = "current_data"

# Combine the data for potential future processing
combined_df = pd.concat([reference_df, current_df], ignore_index=True)

# Prepare data for Evidently
reference_data = reference_df.drop(columns=["Dataset"])
current_data = current_df.drop(columns=["Dataset"])

# Run Evidently's report
print(f"reference_data: {reference_data.columns}")
print(f"current_data: {current_data.columns}")
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_data, current_data=current_data)

# Save the report
report.save_html("data_drift.html")
