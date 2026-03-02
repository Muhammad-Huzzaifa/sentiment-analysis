import kaggle
from pathlib import Path

project_root = Path(__file__).parent.parent
data_raw_dir = project_root / "data" / "raw"

data_raw_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Women's E-Commerce Clothing Reviews dataset...")
kaggle.api.dataset_download_files(
    'nicapotato/womens-ecommerce-clothing-reviews',
    path=str(data_raw_dir),
    unzip=True
)

for file in data_raw_dir.glob("*.csv"):
    if file.name != "reviews.csv":
        file.rename(data_raw_dir / "reviews.csv")
        print(f"Renamed {file.name} to reviews.csv")
        break

print(f"Dataset downloaded successfully to {data_raw_dir / 'reviews.csv'}")
