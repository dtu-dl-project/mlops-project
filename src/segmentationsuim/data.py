import os 
import requests
import zipfile
import gdown

def download_dataset():
    test_url = "https://drive.google.com/file/d/1diN3tNe2nR1eV3Px4gqlp6wp3XuLBwDy/view?usp=drive_link"
    train_url = "https://drive.google.com/file/d/1YWjUODQWwQ3_vKSytqVdF4recqBOEe72/view?usp=drive_link"

    test_ID = test_url.split("/d/")[1].split("/view")[0]
    train_ID = train_url.split("/d/")[1].split("/view")[0]

    data_path_raw = "data/raw"

    files = [
        {"id": test_ID, "name": "test.zip"},
        {"id": train_ID, "name": "train_val.zip"},
    ]

    # Ensure the directory exists
    os.makedirs(data_path_raw, exist_ok=True)

    # Check if the folder is empty (excluding .gitkeep)
    folder_content = [f for f in os.listdir(data_path_raw) if f != ".gitkeep"]

    if not folder_content:
        print("Dataset folder empty, downloading dataset...")

        for file in files:
            file_url = f"https://drive.google.com/uc?id={file['id']}"
            output_path = os.path.join(data_path_raw, file["name"])

            # Download the file
            if not os.path.exists(output_path):
                print(f"Downloading {file['name']} from {file_url}...")
                downloaded = gdown.download(file_url, output_path, quiet=False, fuzzy = True)

                # Check if the file was downloaded successfully
                if downloaded is None:
                    print(f"Failed to download {file['name']}. Skipping...")
                    continue

                print(f"Downloaded {file['name']} to {output_path}. File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

                # Extract if it's a zip file
                if output_path.endswith(".zip"):
                    print(f"Extracting {output_path}...")
                    try:
                        with zipfile.ZipFile(output_path, 'r') as zip_ref:
                            for file_name in zip_ref.namelist():
                                zip_ref.extract(file_name, data_path_raw)
                                print(f"Extracted: {file_name}")
                        print(f"Extracted {file['name']} to {data_path_raw}.")
                    except zipfile.BadZipFile:
                        print(f"Failed to extract {file['name']}. The file may be corrupted.")
                    
                    # Delete the zip file after extraction
                    os.remove(output_path)
                    print(f"Deleted {output_path} after extraction.")
            else:
                print(f"{file['name']} already exists, skipping download.")
        print("Dataset downloaded and extracted successfully.")
    else:
        print("The folder is not empty. Skipping download.")

if __name__ == "__main__":
    download_dataset()
