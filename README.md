# Install Python 3.11 from python.org if you haven't already
# Create new venv with Python 3.11
python3.11 -m venv venv
# Activate the virtual environment
venv\Scripts\activate  
python -m pip install --upgrade pip
pip install -r requirements.txt

# Data Drift Analysis Tool

This tool analyzes data drift between two datasets and calculates feature importance scores. It's particularly useful for monitoring model performance and data quality over time.

## Prerequisites

### Python Version
```
- Python 3.11 or higher recommended
- Created and tested with Python 3.11.9
```

### Environment Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install required packages:
```bash
pip install pandas==2.2.1 numpy==1.26.4 scikit-learn==1.4.1.post1 evidently==0.4.12 shap==0.44.1 ipython==8.22.2
```

Or using the requirements.txt:
```bash
pip install -r requirements.txt
```

## Project Structure
```
FarmersAWSMonitor/
│
├── config/
│   └── config.json
│
├── data/
│   ├── Credit_score_cleaned_data_Aug.csv
│   └── Credit_score_cleaned_data_Sep.csv
│
├── results/
│   └── drift_analysis_results_Sep.csv
│
├── venv/
├── mainscript.py
└── README.md
```

## Configuration

The `config/config.json` should contain:
```json
{
    "reference_data_path": "data/Credit_score_cleaned_data_Aug.csv",
    "current_data_path": "data/Credit_score_cleaned_data_Sep.csv",
    "target": "Credit_Score",
    "drop_columns": ["Customer_ID"],
    "feature_importance_methods": ["random_forest"],
    "drift_tests": {
        "numerical": ["ks"],
        "categorical": ["chisquare"]
    }
}
```

## Running the Analysis

1. Make sure your virtual environment is activated
2. Run the main script:
```bash
python mainscript.py
```

## Output

The script generates a CSV file in the `results` folder containing:
```
- Feature importance scores
- Drift scores
- Normalized importance and drift values
- Priority scores
- Test methods used
- Timestamps
```

## Columns in Output

```
- Feature: Name of the feature
- Timestamp: When the analysis was run
- Feature_Type: numerical or categorical
- Feature_Importance_random_forest: Raw importance score
- Importance_Score_Status: Available/Aggregated
- Drift_Score: Raw drift score
- Drift_Test_Method: Test used (ks/chisquare)
- Normalized_Importance: Scaled importance (0-1)
- Normalized_Drift: Scaled drift score (0-1)
- Priority_Score: Combined importance and drift (0-1)
```

## Troubleshooting

1. If you get "IPython could not be loaded!" warning:
```
- This is normal and won't affect the analysis
- Can be fixed by: pip install ipython
```

2. If you get import errors:
```
- Ensure all requirements are installed
- Check Python version compatibility
- Verify virtual environment is activated
```

3. For data loading issues:
```
- Verify file paths in config.json
- Check CSV file formatting
- Ensure proper column names
```

## Notes
```
- The tool automatically handles both numerical and categorical features
- Drift scores are calculated using KS-test for numerical and Chi-square for categorical features
- Feature importance is calculated using Random Forest by default
```