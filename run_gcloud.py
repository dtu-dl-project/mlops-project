import subprocess
import yaml
import wandb


def main():
    # Log in to wandb and fetch the API key
    wandb.login()
    api_key = wandb.api.api_key

    if not api_key:
        raise ValueError("WANDB_API_KEY could not be retrieved. Make sure you are logged in.")

    # Define the training configuration
    train_config = {
        "workerPoolSpecs": [
            {
                "machineSpec": {"machineType": "n1-highmem-2"},
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": "europe-west1-docker.pkg.dev/massive-mantra-315507/artifacts/train",
                    "env": [{"name": "WANDB_API_KEY", "value": api_key}],
                },
            }
        ]
    }

    # Save the configuration to a YAML file
    config_filename = "train_config.yaml"
    with open(config_filename, "w") as yaml_file:
        yaml.dump(train_config, yaml_file, default_flow_style=False)

    # Construct the gcloud command
    gcloud_command = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        "--region=europe-west1",
        f"--config={config_filename}",
        "--display-name=train",
    ]

    # Execute the gcloud command
    try:
        subprocess.run(gcloud_command, check=True)
        print("Training job started successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to start the training job.", e)


if __name__ == "__main__":
    main()
