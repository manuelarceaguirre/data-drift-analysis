import pandas as pd
import numpy as np
from typing import Optional, List
import os
import shutil

# Define the CSV path here
CSV_PATH = "Credit_score_cleaned_data_Nov.csv"  # Replace with your CSV path
COLUMNS_TO_SKIP = ["Customer_ID"]  # Add any columns you want to skip
BACKUP = True  # Set to False if you don't want to create a backup

def fill_nan_values(
    file_path: str,
    columns_to_skip: Optional[List[str]] = None,
    create_backup: bool = True
) -> pd.DataFrame:
    """
    Fill NaN values in a CSV file using random patterns and replace the original file.
    
    Args:
        file_path: Path to CSV file
        columns_to_skip: List of column names to skip (optional)
        create_backup: Whether to create a backup of the original file
    """
    # Create backup if requested
    if create_backup:
        backup_path = f"{file_path}.backup"
        print(f"\nCreating backup at: {backup_path}")
        shutil.copy2(file_path, backup_path)
    
    # Read the CSV file
    print(f"\nReading file: {file_path}")
    df = pd.read_csv(file_path)
    
    # Initialize counters
    total_nan = df.isna().sum().sum()
    filled_count = 0
    
    print(f"\nTotal NaN values found: {total_nan}")
    if total_nan == 0:
        print("No NaN values found. File remains unchanged.")
        return df
        
    print("\nAnalyzing columns...")
    
    # Store original dtypes
    original_dtypes = df.dtypes.to_dict()
    
    # Process each column
    for column in df.columns:
        if columns_to_skip and column in columns_to_skip:
            print(f"\nSkipping column: {column}")
            continue
            
        nan_count = df[column].isna().sum()
        if nan_count == 0:
            continue
            
        print(f"\nProcessing column: {column}")
        print(f"NaN values found: {nan_count}")
        
        try:
            # Check if column can be converted to numeric
            numeric_series = pd.to_numeric(df[column], errors='coerce')
            is_numeric = not numeric_series.isna().all()
            
            if is_numeric:
                # For numeric columns, use random sampling within mean ± 2*std
                valid_values = df[column].dropna()
                mean_val = valid_values.mean()
                std_val = valid_values.std()
                if pd.isna(std_val) or std_val == 0:
                    # If std is 0 or NaN, just use the mean
                    df[column] = df[column].fillna(mean_val)
                else:
                    # Generate random values within mean ± 2*std
                    random_values = np.random.normal(mean_val, std_val, size=nan_count)
                    df.loc[df[column].isna(), column] = random_values
                
                print(f"Filled with random numeric values (mean: {mean_val:.2f}, std: {std_val:.2f})")
            
            else:
                # For categorical columns, use random sampling from existing values
                valid_values = df[column].dropna().values
                if len(valid_values) > 0:
                    random_values = np.random.choice(valid_values, size=nan_count)
                    df.loc[df[column].isna(), column] = random_values
                    print(f"Filled with random categorical values from existing data")
                else:
                    print(f"Warning: No valid values found in column {column}")
            
            filled_count += nan_count
            
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
            continue
    
    # Restore original dtypes
    for column, dtype in original_dtypes.items():
        try:
            df[column] = df[column].astype(dtype)
        except:
            print(f"Warning: Could not restore original dtype for {column}")
    
    # Save the results back to the original file
    print(f"\nSaving changes to: {file_path}")
    df.to_csv(file_path, index=False)
    
    print(f"\nSummary:")
    print(f"Total NaN values: {total_nan}")
    print(f"Filled values: {filled_count}")
    print(f"Remaining NaN values: {df.isna().sum().sum()}")
    
    if create_backup:
        print(f"\nOriginal file backed up at: {backup_path}")
    print(f"Updated file saved at: {file_path}")
    
    return df

if __name__ == "__main__":
    # Run the function with the defined path
    filled_df = fill_nan_values(
        file_path=CSV_PATH,
        columns_to_skip=COLUMNS_TO_SKIP,
        create_backup=BACKUP
    )
    
    # Print column-wise NaN summary after filling
    print("\nColumn-wise NaN summary after filling:")
    nan_summary = filled_df.isna().sum()
    for column, count in nan_summary[nan_summary > 0].items():
        print(f"{column}: {count} NaN values")
