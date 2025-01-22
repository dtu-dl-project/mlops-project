from locust import HttpUser, TaskSet, task, between
import os
import random


class UserBehavior(TaskSet):
    @task
    def predict(self):
        # Choose a random model type
        model_type = random.choice(["unet", "transformer"])

        # Path to a sample image for testing
        image_path = "tests/test_data/processed/images/00000.png"  # Replace with your actual test image path
        if not os.path.exists(image_path):
            print("Test image not found. Please add a test image to the path specified.")
            return

        # Read the image as binary
        with open(image_path, "rb") as image_file:
            files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}

            # Send the POST request to the /predict/ endpoint
            response = self.client.post(f"/predict/?model_type={model_type}", files=files)

            # Log the response
            if response.status_code == 200:
                print("Prediction successful!")
            else:
                print(f"Prediction failed: {response.status_code}, {response.text}")


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)  # Wait time between tasks (1 to 3 seconds)
    host = "https://api-288862848403.europe-west1.run.app"  # Replace with your FastAPI service URL
