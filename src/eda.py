
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Resolve project paths
    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "raw"
        / "cdc_diabetes_health_indicators.csv"
    )

    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    target = "Diabetes_binary"
    X = df.drop(columns=[target])
    y = df[target]

    print("=" * 60)
    print("DIABETES DATASET: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Basic information
    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # 2. Duplicate records
    duplicate_count = df.duplicated().sum()

    print("\n2. DUPLICATE RECORDS")
    print("-" * 40)
    print("Exact duplicate rows:", duplicate_count)

    # 3. Target distribution
    print("\n3. TARGET DISTRIBUTION")
    print("-" * 40)

    target_counts = y.value_counts().sort_index()
    target_percentages = (
        y.value_counts(normalize=True).sort_index() * 100
    )

    target_summary = pd.DataFrame({
        "Count": target_counts,
        "Percentage": target_percentages.round(2),
    })

    print(target_summary)

    target_summary.to_csv(
        reports_dir / "target_distribution.csv"
    )

    # 4. Feature summary
    print("\n4. FEATURE SUMMARY")
    print("-" * 40)
    print(X.describe().T)

    X.describe().T.to_csv(
        reports_dir / "feature_summary.csv"
    )

    # 5. Class distribution chart
    plt.figure(figsize=(8, 5))

    ax = sns.countplot(
        data=df,
        x=target,
        hue=target,
        order=[0, 1],
        palette="Set2",
        legend=False,
    )

    ax.set_title("Diabetes Target Distribution")
    ax.set_xlabel("Target: 0 = No, 1 = Prediabetes or Diabetes")
    ax.set_ylabel("Number of Records")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "target_distribution.png",
        dpi=150,
    )
    plt.close()

    # 6. BMI distribution by target
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="BMI",
        hue=target,
        bins=40,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
    )

    plt.title("BMI Distribution by Diabetes Target")
    plt.xlabel("BMI")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "bmi_distribution.png",
        dpi=150,
    )
    plt.close()

    # 7. Feature correlations
    # These are associations, not proof of causation.
    correlation = df.corr(numeric_only=True)

    correlation.to_csv(
        reports_dir / "correlation_matrix.csv"
    )

    target_correlations = (
        correlation[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )

    print("\n5. FEATURES MOST ASSOCIATED WITH TARGET")
    print("-" * 40)
    print(target_correlations)

    plt.figure(figsize=(15, 12))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.1,
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "correlation_heatmap.png",
        dpi=150,
    )
    plt.close()

    print("\nEDA completed successfully!")
    print("Reports directory:", reports_dir)
    print("Figures directory:", figures_dir)


if __name__ == "__main__":
    main()
