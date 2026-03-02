import kaggle
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent
data_raw_dir = project_root / "data" / "raw"

# Create directory if it doesn't exist
data_raw_dir.mkdir(parents=True, exist_ok=True)

# Download dataset from Kaggle
print("Downloading Women's E-Commerce Clothing Reviews dataset...")
kaggle.api.dataset_download_files(
    'nicapotato/womens-ecommerce-clothing-reviews',
    path=str(data_raw_dir),
    unzip=True
)

# Rename the downloaded file to reviews.csv
for file in data_raw_dir.glob("*.csv"):
    if file.name != "reviews.csv":
        file.rename(data_raw_dir / "reviews.csv")
        print(f"Renamed {file.name} to reviews.csv")
        break

print(f"Dataset downloaded successfully to {data_raw_dir / 'reviews.csv'}")
