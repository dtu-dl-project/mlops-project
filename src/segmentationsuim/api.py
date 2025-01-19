from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse
from typing import Literal
from torchvision import transforms
from hydra import initialize, compose
from omegaconf import DictConfig
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import uuid

# Import your models
from segmentationsuim.train import UNetModule, unet, Trans

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories for uploads and results
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def load_model(cfg: DictConfig):
    """
    Load the appropriate model based on the configuration file.
    """
    checkpoint_path = os.path.join(cfg.checkpoints.dirpath, cfg.checkpoints.filename)
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


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_type: Literal["unet", "transformer"] = Query(..., description="Choose the model: unet or transformer"),
):
    """
    API endpoint for predicting with UNet or Transformer models.
    - Upload an image.
    - Choose the model type (`unet` or `transformer`).
    """
    # Determine the configuration file to load based on the model type
    config_file = f"{model_type}.yaml"  # unet.yaml or transformer.yaml
    # Dynamically calculate the absolute path to `configs`
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name=config_file)

    # Save the uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Preprocess the uploaded image
    image = Image.open(input_path).convert("RGB")
    image_transform = transforms.Compose(
        [transforms.Resize(cfg.image_transformations.image_size), transforms.ToTensor()]
    )
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    # Load the model
    try:
        model = load_model(cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Predict using the selected model
    with torch.no_grad():
        if cfg.training.model == "unet":
            output = model.unet(img_tensor)
            output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
            prediction = output.argmax(dim=1).squeeze().cpu().numpy()
        elif cfg.training.model == "transformer":
            img_proc = model.processor(images=img_tensor, return_tensors="pt").pixel_values.to(DEVICE)
            logits = model.model(img_proc).logits
            prediction = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Save the prediction as an image
    result_path = os.path.join(RESULT_FOLDER, f"{uuid.uuid4()}_prediction.png")
    plt.imsave(result_path, prediction, cmap="tab20")

    # Return the prediction image
    return FileResponse(result_path, media_type="image/png", filename=f"prediction_{cfg.training.model}.png")
