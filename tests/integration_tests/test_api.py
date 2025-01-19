from fastapi.testclient import TestClient
from segmentationsuim.api import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the API for segment your underwater image. You can choose between two different models: unet and transformer. Upload your image and see the magic!!"
    }
