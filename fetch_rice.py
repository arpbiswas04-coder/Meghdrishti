import urllib.request
import zipfile
from pathlib import Path

# Alternative mirror: the RICE dataset is also available on Kaggle
# Install kaggle CLI and download:
import subprocess

Path("data/RICE_DATASET").mkdir(parents=True, exist_ok=True)

print("Attempting Kaggle download for RICE dataset...")
print("If you have kaggle CLI configured, this will work automatically.")
print("Otherwise download manually from:")
print("  https://www.kaggle.com/datasets/helpman01/rice-dataset")
print("  Extract into: data/RICE_DATASET/")
print()

# Try kaggle CLI
result = subprocess.run(
    ["kaggle", "datasets", "download", "-d", "helpman01/rice-dataset",
     "-p", "data/RICE_DATASET/", "--unzip"],
    capture_output=False,
    shell=True # adding shell=True for windows
)

if result.returncode == 0:
    print("RICE dataset downloaded successfully via Kaggle.")
else:
    print("Kaggle download failed. Manual download required.")
    print("Go to: https://www.kaggle.com/datasets/helpman01/rice-dataset")
    print("Download and extract into data/RICE_DATASET/")
