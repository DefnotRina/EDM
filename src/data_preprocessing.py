import pandas as pd
import os

def load_and_clean_data(file_path):
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv(file_path)
    
    # Drop any immediate NaN values to ensure data quality
    df = df.dropna()
    
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Map categorical values to numerical representations
    diet_map = {'Poor': 0, 'Fair': 0.5, 'Good': 1.0}
    df['diet_num'] = df['diet_quality'].map(diet_map)
    df['dropout_num'] = df['dropout_risk'].map({'No': 0, 'Yes': 1})
    
    # Save the cleaned intermediate data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    
    print("Data cleaning complete. Saved to 'data/processed/cleaned_data.csv'.\n")
    return df
