import os
import pandas as pd
import json
import argparse
from typing import Dict, List

def get_file_path() -> str:
    return input("Enter path to your data file: ")

def display_columns(df: pd.DataFrame) -> None:
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

def get_drop_columns(df: pd.DataFrame) -> List[str]:
    while True:
        print("\nOptions:")
        print("1. Drop columns")
        print("2. Continue with analysis")
        
        choice = input("\nSelect option (1-2): ")
        if choice == "1":
            display_columns(df)
            cols_to_drop = input("\nEnter column numbers to drop (comma-separated) or press Enter to skip: ")
            if cols_to_drop.strip():
                return [df.columns[int(i)-1] for i in cols_to_drop.split(",")]
        elif choice == "2":
            return []

def get_time_unit_column(df: pd.DataFrame) -> str:
    print("\nPlease specify the time unit column (if any) from the remaining columns.")
    print("If there is no time unit column, just press Enter.\n")
    display_columns(df)
    
    choice = input("\nEnter the number corresponding to the time unit column or press Enter to skip: ")
    if choice.strip():
        return df.columns[int(choice)-1]
    return None

def get_target_variable(df: pd.DataFrame) -> str:
    print("\nSelect target variable:")
    display_columns(df)
    
    choice = input("\nEnter number: ")
    return df.columns[int(choice)-1]

def get_feature_importance_methods() -> List[str]:
    methods = []
    print("\nFeature Importance Methods:")
    print("1. Random Forest (faster, built-in importance)")
    print("2. Permutation Importance (slower, model agnostic)")
    print("3. SHAP Values (slower, more detailed)")
    print("4. Mutual Information (faster, statistical measure)")
    print("5. Done selecting")
    
    while True:
        choice = input("\nSelect a method (1-5): ")
        if choice == "5":
            break
        if choice in ["1", "2", "3", "4"]:
            method_map = {
                "1": "random_forest",
                "2": "permutation",
                "3": "shap",
                "4": "mutual_info"
            }
            methods.append(method_map[choice])
    return methods

def get_drift_tests(df: pd.DataFrame) -> Dict:
    column_tests = {}
    
    print("\nConfiguring drift tests for each column...")
    for col in df.columns:
        if col in df.select_dtypes(include=['int64', 'float64']).columns:
            print(f"\nColumn: {col}")
            print("Available tests for this numerical column:")
            print("1. Kolmogorov-Smirnov test (ks)")
            print("2. Wasserstein Distance (wasserstein)")
            print("3. Population Stability Index (psi)")
            print("4. Skip this column")
            
            choice = input("Select test (1-4): ")
            if choice != "4":
                test_map = {"1": "ks", "2": "wasserstein", "3": "psi"}
                column_tests[col] = {
                    "type": "numerical",
                    "tests": [test_map[choice]]
                }
        else:
            print(f"\nColumn: {col}")
            print("Available tests for this categorical column:")
            print("1. Chi-square test (chisquare)")
            print("2. Population Stability Index (psi)")
            print("3. Jensen-Shannon Distance (jensenshannon)")
            print("4. Skip this column")
            
            choice = input("Select test (1-4): ")
            if choice != "4":
                test_map = {"1": "chisquare", "2": "psi", "3": "jensenshannon"}
                column_tests[col] = {
                    "type": "categorical",
                    "tests": [test_map[choice]]
                }
    
    return column_tests

def analyze_dataframe(df: pd.DataFrame, reference_path: str, current_path: str) -> Dict:
    # Get all column names
    all_columns = df.columns.tolist()
    
    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Create individual column drift test configurations
    column_drift_tests = {}
    for col in numeric_columns:
        column_drift_tests[col] = {
            "type": "numerical",
            "tests": ["ks"]  # Default to KS test for numerical
        }
    
    for col in categorical_columns:
        column_drift_tests[col] = {
            "type": "categorical",
            "tests": ["chisquare"]  # Default to chi-square for categorical
        }
    
    config = {
        "reference_data_path": reference_path,
        "current_data_path": current_path,
        "target": None,  # Will be set later
        "drop_columns": [],
        "feature_importance_methods": ["random_forest"],
        "available_feature_importance_methods": ["random_forest", "permutation", "shap", "mutual_info"],
        "drift_tests": {
            "default": {
                "numerical": ["ks"],
                "categorical": ["chisquare"]
            },
            "columns": column_drift_tests,
            "available_drift_tests": {
                "numerical": ["ks", "wasserstein", "anderson", "psi"],
                "categorical": ["chisquare", "psi", "jensenshannon"]
            }
        },
        "time_unit_column": None,  # Will be set later
        "results_json_path": "analysis_results.json",
        "output_csv_path": "feature_analysis.csv"
    }
    
    return config

def main():
    # Get reference data path
    reference_path = input("Enter path to your reference data file: ")
    
    # Get current data path
    current_path = input("Enter path to your current data file (or press Enter to skip): ")
    
    try:
        # Read the reference data file
        df = pd.read_csv(reference_path)
        
        # Generate initial config
        config = analyze_dataframe(df, reference_path, current_path)
        
        # Handle drop columns
        print("\nOptions:")
        print("1. Drop columns")
        print("2. Continue with analysis")
        
        choice = input("\nSelect option (1-2): ")
        if choice == "1":
            display_columns(df)
            cols_to_drop = input("\nEnter column numbers to drop (comma-separated) or press Enter to skip: ")
            if cols_to_drop.strip():
                drop_cols = [df.columns[int(i)-1] for i in cols_to_drop.split(",")]
                config["drop_columns"] = drop_cols
        
        # Get time unit column
        print("\nPlease specify the time unit column (if any) from the remaining columns.")
        print("If there is no time unit column, just press Enter.\n")
        display_columns(df)
        time_col = input("\nEnter the number corresponding to the time unit column or press Enter to skip: ")
        if time_col.strip():
            config["time_unit_column"] = df.columns[int(time_col)-1]
        
        # Get target variable
        print("\nSelect target variable:")
        display_columns(df)
        target = input("\nEnter number: ")
        config["target"] = df.columns[int(target)-1]
        
        # Get feature importance methods
        config["feature_importance_methods"] = get_feature_importance_methods()
        
        # Get drift tests for each column
        config["drift_tests"]["columns"] = get_drift_tests(df)
        
        # Create output directory if it doesn't exist
        os.makedirs("config", exist_ok=True)
        
        # Save config
        with open("config/config.json", 'w') as f:
            json.dump(config, f, indent=4)
            
        print("\nConfiguration file generated successfully: config/config.json")
        
    except Exception as e:
        print(f"Error generating config: {str(e)}")

if __name__ == "__main__":
    main() 