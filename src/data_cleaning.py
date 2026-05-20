import os
import numpy as np
import pandas as pd

# For MICE: to Impute Missing values
from sklearn.impute import IterativeImputer, KNNImputer

# IsolationForest: For Outlier Detection
from sklearn.ensemble import IsolationForest

# Z-Score based outlier detection
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

from src.config import PROCESSED_DATA_DIR


class DataCleaner:
    """
    A pipeline class for cleaning tabular datasets.

    Steps:
        1. load_data()              - Load CSV and identify numerical columns
        2. imputing_missing_values()- Fill missing values using MICE or KNN Imputer
        3. detect_outliers()        - Flag (and optionally remove) outliers via Isolation Forest / IQR / Z-Score
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

    def _ensure_no_nulls(self) -> None:
        """Raise a clear error if NaN values are still present — must impute first."""
        if self.df[self.numerical_cols].isnull().any().any():
            raise RuntimeError(
                "NaN values found in numerical columns. "
                "Run imputing_missing_values() before detect_outliers()."
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

        # Exclude outlier flag column if it exists from a previous run
        self.numerical_cols = [
            col for col in self.numerical_cols if col != "is_outlier"
        ]

        if len(self.numerical_cols) == 0:
            raise ValueError(
                "No numerical columns found for processing. "
                "Verify the dataset or the _id exclusion filter."
            )

        print(f"Loaded data shape      : {self.df.shape}")
        print(f"Numerical cols (non-id): {len(self.numerical_cols)}")

    def imputing_missing_values(
        self,
        strategy: str = "mice",
        min_value: float | None = None,
    ) -> None:
        """
        Fill missing numerical values using MICE or KNN Imputer.

        Parameters
        ----------
        strategy : str
            'mice' uses IterativeImputer (default), 'knn' uses KNNImputer.
            MICE is more accurate; KNN is faster on smaller datasets.
        min_value : float | None
            Lower bound for imputed values — only applies to MICE.
            Default is None (no clipping). Pass 0.0 only if ALL numerical
            columns are guaranteed non-negative (e.g. price, freight).
            Do NOT use 0.0 if lat/lng or delay columns are present.
        """
        self._ensure_loaded()

        if strategy not in ("mice", "knn"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use 'mice' or 'knn'."
            )

        print(f"\nStarting Imputation (strategy={strategy})...")

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

        # Select imputer based on strategy
        if strategy == "mice":
            imputer = IterativeImputer(
                max_iter=10,
                tol=1e-3,           # Stop early if converged — helps with wide-range columns
                random_state=42,
                min_value=min_value,
            )
        else:
            # KNN Imputer — does not support min_value, handles it naturally
            imputer = KNNImputer(n_neighbors=5)

        imputed_data = imputer.fit_transform(self.df[self.numerical_cols])

        # Replace original columns with imputed values
        self.df[self.numerical_cols] = imputed_data

        print(f"Imputation complete (strategy={strategy}).")

    def detect_outliers(
        self,
        method: str = "isolation_forest",
        contamination_rate: float = 0.02,
        remove_outliers: bool = False,
    ) -> None:
        """
        Flag (and optionally remove) outliers.

        Parameters
        ----------
        method : str
            'isolation_forest' (default) — tree-based, works well on high-dimensional data.
            'iqr'              — interquartile range, good for univariate outliers.
            'zscore'           — standard deviation based, assumes normal distribution.
        contamination_rate : float
            Expected proportion of outliers (0 < value <= 0.5).
            Only used by isolation_forest method.
        remove_outliers : bool
            If True, outlier rows are dropped from self.df.
            If False (default), an ``is_outlier`` flag column is added instead.
        """
        self._ensure_loaded()
        self._ensure_no_nulls()   # NaN check — imputation must run first

        if method not in ("isolation_forest", "iqr", "zscore"):
            raise ValueError(
                f"Unknown method '{method}'. Use 'isolation_forest', 'iqr', or 'zscore'."
            )

        print(f"\nStarting Outlier Detection (method={method})...")

        # --- Isolation Forest ---
        if method == "isolation_forest":

            # Validate contamination rate before passing it to sklearn
            if not (0 < contamination_rate <= 0.5):
                raise ValueError(
                    f"contamination_rate must be between 0 (exclusive) and 0.5 (inclusive), "
                    f"got {contamination_rate}."
                )

            iso_forest = IsolationForest(
                n_estimators=100,
                contamination=contamination_rate,
                random_state=42,
            )
            # -1 => outlier, 1 => normal
            preds = iso_forest.fit_predict(self.df[self.numerical_cols])

        # --- IQR Method ---
        elif method == "iqr":
            Q1 = self.df[self.numerical_cols].quantile(0.25)
            Q3 = self.df[self.numerical_cols].quantile(0.75)
            IQR = Q3 - Q1

            # A row is an outlier if ANY column falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            outlier_mask = (
                (self.df[self.numerical_cols] < (Q1 - 1.5 * IQR)) |
                (self.df[self.numerical_cols] > (Q3 + 1.5 * IQR))
            ).any(axis=1)

            # Map to -1 / 1 to keep same convention as Isolation Forest
            preds = outlier_mask.map({True: -1, False: 1}).values

        # --- Z-Score Method ---
        else:
            z_scores = np.abs(stats.zscore(self.df[self.numerical_cols]))

            # A row is an outlier if ANY column has |z| > 3
            preds = np.where((z_scores > 3).any(axis=1), -1, 1)

        outlier_count = (preds == -1).sum()
        print(f"Flagged {outlier_count} records as outliers "
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
        Note: 'is_outlier' flag column is excluded from the saved file
        to prevent it from interfering with downstream feature engineering.

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

        # Drop is_outlier before saving — prevents it from polluting Day 3 feature engineering
        cols_to_drop = ["is_outlier"] if "is_outlier" in self.df.columns else []
        save_df = self.df.drop(columns=cols_to_drop)

        output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
        save_df.to_csv(output_path, index=False)

        print(f"\nCleaned dataset saved at : {output_path}")
        print(f"Final shape              : {save_df.shape}")


# Orchestrator

if __name__ == "__main__":
    try:
        cleaner = DataCleaner()

        cleaner.load_data()
        cleaner.imputing_missing_values(strategy="mice", min_value=None)
        cleaner.detect_outliers(method="isolation_forest", contamination_rate=0.02, remove_outliers=False)
        cleaner.save_clean_data()

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {exc}")
        raise