import os
import numpy as np
import pandas as pd

# For MICE: to Impute Missing values
from sklearn.impute import IterativeImputer

# IsolationForest: For Outlier Detection
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings('ignore')

from src.config import PROCESSED_DATA_DIR


class DataCleaner:
    """
    A pipeline class for cleaning tabular datasets.

    Steps:
        1. load_data()              - Load CSV and identify numerical columns
        2. imputing_missing_values()- Fill missing values using MICE (Iterative Imputer)
        3. detect_outliers()        - Flag (and optionally remove) outliers via Isolation Forest
        4. save_clean_data()        - Persist the cleaned DataFrame to disk
    """

    def __init__(self, filename: str = "master_dataset.csv"):
        self.filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        self.df: pd.DataFrame | None = None
        self.numerical_cols: list[str] | None = None

    # Internal helpers

    def _ensure_loaded(self) -> None:
        """Raise a clear error if load_data() has not been called yet."""
        if self.df is None:
            raise RuntimeError(
                "Data not loaded. Call load_data() before running this step."
            )

    
    # Pipeline steps

    def load_data(self) -> None:
        """Load the master CSV and identify numerical columns for processing."""
        print(f"Loading data from {self.filepath}")

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.df = pd.read_csv(self.filepath)

        # Basic sanity checks
        if self.df.empty:
            raise ValueError("Loaded DataFrame is empty. Check the source file.")

        # Extract numerical columns only
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude ID columns — they should not participate in statistical operations
        self.numerical_cols = [
            col for col in self.numerical_cols if not col.endswith("_id")
        ]

        if len(self.numerical_cols) == 0:
            raise ValueError(
                "No numerical columns found for processing. "
                "Verify the dataset or the _id exclusion filter."
            )

        print(f"Loaded data shape      : {self.df.shape}")
        print(f"Numerical cols (non-id): {len(self.numerical_cols)}")

    def imputing_missing_values(self, min_value: float | None = 0) -> None:
        """
        Fill missing numerical values using MICE (Iterative Imputer).

        Parameters
        ----------
        min_value : float | None
            Lower bound for imputed values (default 0).
            Pass ``None`` to disable clipping entirely.
        """
        self._ensure_loaded()

        print("\nStarting MICE Imputation...")

        # Log per-column missing counts before imputing
        missing_by_col = self.df[self.numerical_cols].isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]

        total_missing = missing_by_col.sum()

        if total_missing == 0:
            print("No missing values found in numerical columns. Skipping imputation.")
            return

        print(f"Total missing cells    : {total_missing}")
        print("Missing values per column:")
        print(missing_by_col.to_string())

        # MICE via IterativeImputer
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            min_value=min_value,   # Prevents negative / out-of-range imputed values
        )
        imputed_data = imputer.fit_transform(self.df[self.numerical_cols])

        # Replace original columns with imputed values
        self.df[self.numerical_cols] = imputed_data

        print("MICE Imputation complete.")

    def detect_outliers(
        self,
        contamination_rate: float = 0.02,
        remove_outliers: bool = False,
    ) -> None:
        """
        Flag (and optionally remove) outliers using Isolation Forest.

        Parameters
        ----------
        contamination_rate : float
            Expected proportion of outliers in the data (0 < value <= 0.5).
        remove_outliers : bool
            If True, outlier rows are dropped from self.df.
            If False (default), an ``is_outlier`` flag column is added instead.
        """
        self._ensure_loaded()

        # Validate contamination rate before passing it to sklearn
        if not (0 < contamination_rate <= 0.5):
            raise ValueError(
                f"contamination_rate must be between 0 (exclusive) and 0.5 (inclusive), "
                f"got {contamination_rate}."
            )

        print("\nStarting Isolation Forest Outlier Detection...")

        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination_rate,
            random_state=42,
        )

        # -1 => outlier, 1 => normal
        preds = iso_forest.fit_predict(self.df[self.numerical_cols])
        outlier_count = (preds == -1).sum()

        print(f"Isolation Forest flagged {outlier_count} records as outliers "
              f"({outlier_count / len(self.df) * 100:.2f}% of total).")

        if remove_outliers:
            before = len(self.df)
            self.df = self.df[preds == 1].reset_index(drop=True)
            after = len(self.df)
            print(f"Removed {before - after} outlier rows. "
                  f"Remaining rows: {after}.")
        else:
            self.df["is_outlier"] = preds
            print("Outlier flag column 'is_outlier' added (-1 = outlier, 1 = normal).")
            print("Outlier rows are still present in the dataset.")

    def save_clean_data(self, output_filename: str = "cleaned_master_dataset.csv") -> None:
        """
        Save the cleaned DataFrame to the processed data directory.

        Parameters
        ----------
        output_filename : str
            Name of the output CSV file.
        """
        self._ensure_loaded()

        # Warn if outlier rows will be included in the output
        if "is_outlier" in self.df.columns and (self.df["is_outlier"] == -1).any():
            outlier_rows = (self.df["is_outlier"] == -1).sum()
            print(
                f"\nWarning: {outlier_rows} flagged outlier rows are included "
                "in the saved file. Pass remove_outliers=True to detect_outliers() "
                "if you want them excluded."
            )

        output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
        self.df.to_csv(output_path, index=False)

        print(f"\nCleaned dataset saved at : {output_path}")
        print(f"Final shape              : {self.df.shape}")


# Orchestrator

if __name__ == "__main__":
    try:
        cleaner = DataCleaner()

        cleaner.load_data()
        cleaner.imputing_missing_values(min_value=0)
        cleaner.detect_outliers(contamination_rate=0.02, remove_outliers=False)
        cleaner.save_clean_data()

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {exc}")
        raise