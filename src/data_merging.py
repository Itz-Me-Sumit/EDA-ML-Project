import os
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class OListDataProcessor():

    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.files = {
            "orders":      "olist_orders_dataset.csv",
            "items":       "olist_order_items_dataset.csv",
            "products":    "olist_products_dataset.csv",
            "customers":   "olist_customers_dataset.csv",
            "payments":    "olist_order_payments_dataset.csv",
            "reviews":     "olist_order_reviews_dataset.csv",
            "sellers":     "olist_sellers_dataset.csv",
            "translation": "product_category_name_translation.csv",
            "geolocation": "olist_geolocation_dataset.csv",   # Geolocation added
        }
        self.dataframes = {}
        self.master_df = None

    def load_raw_data(self):
        print("Loading raw data files...")
        for name, filename in self.files.items():
            filepath = os.path.join(self.raw_dir, filename)
            if os.path.exists(filepath):
                self.dataframes[name] = pd.read_csv(filepath)
            else:
                raise FileNotFoundError(f"Missing file: {filename} in {self.raw_dir}")
        print("All files loaded successfully!")

    def _merge_geolocation(self) -> None:
        """
        Geolocation table mein ek zip code ke multiple lat/lng hote hain.
        Median lete hain — outlier-resistant hota hai.
        Customer aur seller dono ke liye alag join karte hain.
        """
        geo = (
            self.dataframes["geolocation"]
            .groupby("geolocation_zip_code_prefix")
            .agg(
                lat=("geolocation_lat", "median"),
                lng=("geolocation_lng", "median"),
            )
            .reset_index()
        )

        # Customer coordinates
        self.master_df = pd.merge(
            self.master_df,
            geo.rename(columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "lat": "customer_zip_lat",  # Synced with feature engineering
                "lng": "customer_zip_lng",  # Synced with feature engineering
            }),
            on="customer_zip_code_prefix",
            how="left",
        )

        # Seller coordinates
        self.master_df = pd.merge(
            self.master_df,
            geo.rename(columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "lat": "seller_zip_lat",    # Synced with feature engineering
                "lng": "seller_zip_lng",    # Synced with feature engineering
            }),
            on="seller_zip_code_prefix",
            how="left",
        )

        # Print statement using the new column names to avoid KeyError
        print(f"  Geolocation merged ✓  "
              f"(customer nulls: {self.master_df['customer_zip_lat'].isna().sum()}, "
              f"seller nulls: {self.master_df['seller_zip_lat'].isna().sum()})")

    def merge_data(self):
        if not self.dataframes:
            raise ValueError("DataFrame is empty, run load_raw_data() first")
        print("Starting data merging process...")

        self.master_df = self.dataframes["orders"].copy()

        self.master_df = pd.merge(self.master_df, self.dataframes["customers"],   on="customer_id",              how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["items"],       on="order_id",                 how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["products"],    on="product_id",               how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["translation"], on="product_category_name",    how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["payments"],    on="order_id",                 how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["reviews"],     on="order_id",                 how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["sellers"],     on="seller_id",                how="left")

        # Merge geolocation data
        self._merge_geolocation()

        print("Data merging complete!")

    def save_master_dataset(self, filename="master_dataset.csv"):
        if self.master_df is None:
            raise ValueError("Master DF is empty, run merge_data() first")
        output_path = os.path.join(self.processed_dir, filename)
        self.master_df.to_csv(output_path, index=False)
        print(f"Master dataset saved at: {output_path}")
        print(f"Final dataset shape: {self.master_df.shape}")


if __name__ == "__main__":
    processor = OListDataProcessor()
    processor.load_raw_data()
    processor.merge_data()
    processor.save_master_dataset()