import os
import pandas as pd
from src.config import  RAW_DATA_DIR , PROCESSED_DATA_DIR

class OListDataProcessor():

    def __init__(self):
        # Encapsulation
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.files = {
            "orders": "olist_orders_dataset.csv",
            "items": "olist_order_items_dataset.csv",
            "products": "olist_products_dataset.csv",
            "customers": "olist_customers_dataset.csv",
            "payments": "olist_order_payments_dataset.csv",
            "reviews": "olist_order_reviews_dataset.csv",
            "sellers": "olist_sellers_dataset.csv",
            "translation": "product_category_name_translation.csv"
        }
        self.dataframes ={} # Data will Stored here after Loading
        self.master_df = None # Final Merged Data

    def load_raw_data(self):
        # Loads all CSV files into Dictionary
        print("Loading raw data files...")
        
        for name , filename in self.files.items():
            filepath = os.path.join(self.raw_dir , filename)
            if os.path.exists(filepath):
                self.dataframes[name] = pd.read_csv(filepath)
            else:
                raise FileNotFoundError(f"Missing file : {filename} in {self.raw_dir}")
        print("All files loaded successfully !")
    
    def merge_data(self):
        # merging dataframe after left join
        if not self.dataframes:
            raise ValueError("DataFrame is Empty , run load_raw_data() first")
        print("Starting data merging process...")
        
        # Base Table
        self.master_df = self.dataframes["orders"].copy()

        # Applying Left Joins
        self.master_df = pd.merge(self.master_df, self.dataframes["customers"], on="customer_id", how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["items"], on="order_id", how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["products"], on="product_id", how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["translation"], on="product_category_name", how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["payments"], on="order_id", how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["reviews"], on="order_id", how="left")
        self.master_df = pd.merge(self.master_df, self.dataframes["sellers"], on="seller_id", how="left")

        print("Data merging complete !")

    def save_master_dataset(self , filename="master_dataset.csv"):
        # save final processed dataset to folder
        if self.master_df is None:
            raise ValueError("Master DF is empty , run merge_data() first")
        output_path = os.path.join(self.processed_dir,filename)
        self.master_df.to_csv(output_path,index=False)
        print(f"Master dataset successfully saved at: {output_path}")
        print(f"Final dataset Shape: {self.master_df.shape}")

# Orchestrator
if __name__=="__main__":
    # Object Creation
    processor = OListDataProcessor()

    # Method calls
    processor.load_raw_data()
    processor.merge_data()
    processor.save_master_dataset()