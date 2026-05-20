import os
import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR

import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Geospatial & Temporal Feature Engineering pipeline for the Olist dataset.

    Pipeline Steps:
        1. load_data()                    - Load cleaned_master_dataset.csv
        2. engineer_geospatial_features() - Haversine distance + delivery zone flag
        3. engineer_temporal_features()   - Cyclical encoding, delays, time-decay
        4. drop_raw_columns()             - Remove columns used only for engineering
        5. save_featured_data()           - Persist to processed/featured_dataset.csv
    """

    # Earth radius in km — used in Haversine formula
    EARTH_RADIUS_KM = 6371.0

    def __init__(self, filename: str = "cleaned_master_dataset.csv"):
        self.filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        self.df: pd.DataFrame | None = None

        # Timestamp columns we'll parse — defined once, used everywhere
        self.timestamp_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

    # ─────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self.df is None:
            raise RuntimeError(
                "Data not loaded. Call load_data() before running this step."
            )

    @staticmethod
    def _haversine_vectorized(
        lat1: pd.Series,
        lon1: pd.Series,
        lat2: pd.Series,
        lon2: pd.Series,
        earth_radius: float = 6371.0,
    ) -> pd.Series:
        """
        Vectorized Haversine formula — calculates great-circle distance (km)
        between two sets of lat/lon coordinates.

        Formula:
            a = sin²(Δlat/2) + cos(lat1) · cos(lat2) · sin²(Δlon/2)
            c = 2 · atan2(√a, √(1−a))
            d = R · c
        """
        # Convert degrees → radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        # Core haversine calculation
        a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return earth_radius * c

    @staticmethod
    def _cyclical_encode(series: pd.Series, max_val: int) -> tuple[pd.Series, pd.Series]:
        """
        Encode a cyclical feature using sin/cos transformation.

        Why sin/cos?
            A raw month column treats Dec(12) and Jan(1) as far apart.
            But cyclically they're adjacent. Sin/cos preserves this circular nature.

            sin_encoding = sin(2π × value / max_value)
            cos_encoding = cos(2π × value / max_value)

        Returns (sin_series, cos_series)
        """
        angle = 2 * np.pi * series / max_val
        return np.sin(angle), np.cos(angle)

    # ─────────────────────────────────────────────
    # Pipeline Steps
    # ─────────────────────────────────────────────

    def load_data(self) -> None:
        """Load cleaned dataset and parse all timestamp columns upfront."""
        print(f"Loading data from {self.filepath}")

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.df = pd.read_csv(self.filepath)

        if self.df.empty:
            raise ValueError("Loaded DataFrame is empty.")

        # Parse timestamps — do this once here, not repeatedly in each method
        for col in self.timestamp_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        print(f"Loaded shape       : {self.df.shape}")
        print(f"Parsed timestamps  : {[c for c in self.timestamp_cols if c in self.df.columns]}")

    def engineer_geospatial_features(self) -> None:
        """
        Create distance and zone features from lat/lon coordinates.

        New columns created:
            customer_seller_distance_km  — Haversine distance between customer and seller
            is_same_state                — 1 if customer and seller are in the same state
            is_long_distance             — 1 if distance > 1000 km (rough inter-region flag)
        """
        self._ensure_loaded()
        print("\nEngineering geospatial features...")

        # Required coordinate columns
        geo_cols = [
            "customer_zip_code_prefix",   # we use lat/lng from geolocation
            "geolocation_lat",
            "geolocation_lng",
            "seller_zip_code_prefix",
        ]

        # Graceful fallback — Olist dataset has geolocation in a separate table.
        # After Day 1 merge, lat/lng may appear as customer_lat/seller_lat depending
        # on how geolocation was merged. We handle both naming conventions.
        customer_lat = self._resolve_col("customer_zip_lat", "geolocation_lat")
        customer_lon = self._resolve_col("customer_zip_lng", "geolocation_lng")
        seller_lat   = self._resolve_col("seller_zip_lat", "seller_lat")
        seller_lon   = self._resolve_col("seller_zip_lng", "seller_lng")

        if all(c is not None for c in [customer_lat, customer_lon, seller_lat, seller_lon]):
            self.df["customer_seller_distance_km"] = self._haversine_vectorized(
                self.df[customer_lat],
                self.df[customer_lon],
                self.df[seller_lat],
                self.df[seller_lon],
                earth_radius=self.EARTH_RADIUS_KM,
            ).round(2)

            # Long-distance flag — orders > 1000km likely cross multiple states
            self.df["is_long_distance"] = (
                self.df["customer_seller_distance_km"] > 1000
            ).astype(int)

            print(f"  customer_seller_distance_km : created ✓")
            print(f"  is_long_distance            : created ✓")
        else:
            print("  WARNING: lat/lng columns not found. Skipping distance features.")
            print("  Tip: Merge geolocation dataset in data_merging.py first.")

        # Same-state flag — can be created even without lat/lng
        if "customer_state" in self.df.columns and "seller_state" in self.df.columns:
            self.df["is_same_state"] = (
                self.df["customer_state"] == self.df["seller_state"]
            ).astype(int)
            print(f"  is_same_state               : created ✓")

    def _resolve_col(self, *candidates: str) -> str | None:
        """Return the first column name from candidates that exists in df."""
        for col in candidates:
            if col in self.df.columns:
                return col
        return None

    def engineer_temporal_features(self) -> None:
        """
        Extract and encode time-based features from timestamp columns.

        New columns created:
            purchase_hour_sin/cos       — Cyclical encoding of purchase hour
            purchase_dayofweek_sin/cos  — Cyclical encoding of day of week
            purchase_month_sin/cos      — Cyclical encoding of month
            purchase_quarter            — Quarter of year (1–4)
            is_weekend_purchase         — 1 if order placed on Sat/Sun
            approval_delay_hrs          — Hours between purchase and payment approval
            carrier_pickup_delay_hrs    — Hours between approval and carrier pickup
            actual_delivery_days        — Days from purchase to actual delivery
            estimated_delivery_days     — Days from purchase to estimated delivery
            delivery_delay_days         — Positive = late, Negative = early
            is_late_delivery            — Binary flag: 1 if delivered after estimate
            time_decay_weight           — Recency weight (recent orders score higher)
        """
        self._ensure_loaded()
        print("\nEngineering temporal features...")

        purchase_col = "order_purchase_timestamp"

        if purchase_col not in self.df.columns:
            raise RuntimeError(
                f"Column '{purchase_col}' not found. "
                "Ensure timestamp parsing succeeded in load_data()."
            )

        ts = self.df[purchase_col]

        # ── Cyclical Time Features ──────────────────────────────────────────
        sin_hour, cos_hour = self._cyclical_encode(ts.dt.hour, max_val=24)
        self.df["purchase_hour_sin"] = sin_hour.round(6)
        self.df["purchase_hour_cos"] = cos_hour.round(6)

        sin_dow, cos_dow = self._cyclical_encode(ts.dt.dayofweek, max_val=7)
        self.df["purchase_dayofweek_sin"] = sin_dow.round(6)
        self.df["purchase_dayofweek_cos"] = cos_dow.round(6)

        sin_month, cos_month = self._cyclical_encode(ts.dt.month, max_val=12)
        self.df["purchase_month_sin"] = sin_month.round(6)
        self.df["purchase_month_cos"] = cos_month.round(6)

        self.df["purchase_quarter"] = ts.dt.quarter
        self.df["is_weekend_purchase"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

        print("  Cyclical features (hour, dayofweek, month) : created ✓")
        print("  purchase_quarter, is_weekend_purchase      : created ✓")

        # ── Delay / Duration Features ───────────────────────────────────────
        approved_col   = "order_approved_at"
        carrier_col    = "order_delivered_carrier_date"
        delivered_col  = "order_delivered_customer_date"
        estimated_col  = "order_estimated_delivery_date"

        # Hours between purchase and payment approval
        if approved_col in self.df.columns:
            self.df["approval_delay_hrs"] = (
                (self.df[approved_col] - ts)
                .dt.total_seconds()
                .div(3600)
                .round(2)
            )
            print("  approval_delay_hrs                         : created ✓")

        # Hours from approval to carrier pickup
        if approved_col in self.df.columns and carrier_col in self.df.columns:
            self.df["carrier_pickup_delay_hrs"] = (
                (self.df[carrier_col] - self.df[approved_col])
                .dt.total_seconds()
                .div(3600)
                .round(2)
            )
            print("  carrier_pickup_delay_hrs                   : created ✓")

        # Actual vs estimated delivery — the most important delay feature
        if delivered_col in self.df.columns and estimated_col in self.df.columns:
            self.df["actual_delivery_days"] = (
                (self.df[delivered_col] - ts)
                .dt.total_seconds()
                .div(86400)   # seconds → days
                .round(2)
            )
            self.df["estimated_delivery_days"] = (
                (self.df[estimated_col] - ts)
                .dt.total_seconds()
                .div(86400)
                .round(2)
            )
            # Positive = late, Negative = early delivery
            self.df["delivery_delay_days"] = (
                self.df["actual_delivery_days"] - self.df["estimated_delivery_days"]
            ).round(2)

            self.df["is_late_delivery"] = (
                self.df["delivery_delay_days"] > 0
            ).astype(int)

            print("  actual/estimated_delivery_days             : created ✓")
            print("  delivery_delay_days, is_late_delivery      : created ✓")

        # ── Time-Decay Feature ───────────────────────────────────────────────
        # Recent orders are weighted higher — useful for time-aware models.
        # Formula: exp(-λ × days_since_order), λ = 0.001 (slow decay)
        reference_date = ts.max()
        days_since = (reference_date - ts).dt.days
        self.df["time_decay_weight"] = np.exp(-0.001 * days_since).round(6)
        print("  time_decay_weight                          : created ✓")

    def drop_raw_columns(self) -> None:
        """
        Drop raw timestamp columns after feature extraction.
        These columns have served their purpose and would confuse tree-based models
        if left as raw datetime objects.
        """
        self._ensure_loaded()

        cols_to_drop = [col for col in self.timestamp_cols if col in self.df.columns]

        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f"\nDropped raw timestamp columns: {cols_to_drop}")
        else:
            print("\nNo timestamp columns to drop.")

    def save_featured_data(
        self, output_filename: str = "featured_dataset.csv"
    ) -> None:
        """Save the feature-engineered dataset to processed directory."""
        self._ensure_loaded()

        output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
        self.df.to_csv(output_path, index=False)

        print(f"\nFeatured dataset saved at : {output_path}")
        print(f"Final shape               : {self.df.shape}")

        # Print a summary of new feature columns
        feature_cols = [
            c for c in self.df.columns
            if any(tag in c for tag in [
                "distance", "delay", "delivery", "purchase_",
                "is_late", "is_long", "is_same", "is_weekend",
                "decay", "quarter", "approval", "carrier",
            ])
        ]
        print(f"\nEngineered features ({len(feature_cols)} total):")
        for col in feature_cols:
            non_null = self.df[col].notna().sum()
            print(f"  {col:<40} non-null: {non_null}")


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        engineer = FeatureEngineer()

        engineer.load_data()
        engineer.engineer_geospatial_features()
        engineer.engineer_temporal_features()
        engineer.drop_raw_columns()
        engineer.save_featured_data()

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed: {exc}")
        raise