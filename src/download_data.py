
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def main():
    # Project directories
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading the UCI CDC Diabetes Health Indicators dataset...")

    # Fetch dataset ID 891 from UCI
    dataset = fetch_ucirepo(id=891)

    # Extract features and target
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    print("\nFeatures shape:", features.shape)
    print("Target shape:", targets.shape)

    # Combine features and target
    df = pd.concat([features, targets], axis=1)

    # Save the downloaded dataset
    output_path = raw_data_dir / "cdc_diabetes_health_indicators.csv"
    df.to_csv(output_path, index=False)

    print("\nDataset downloaded successfully!")
    print("Saved to:", output_path)
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nTarget distribution:")
    for column in targets.columns:
        print(df[column].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
