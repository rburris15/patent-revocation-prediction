# Patent Revocation Prediction

## Overview
This project is designed for processing, analyzing, and extracting insights from EPO data. It includes modules for exploratory data analysis (EDA), preprocessing, and feature selection, as well as utilities for interacting with EPO (European Patent Office) data. The project is structured to facilitate reproducible and scalable data workflows.

## Repository Structure
```
│   ├── 1_EDA.py                                   # Script to visualize and explore raw data
│   ├── 2a_topfeat_preprocessing.py            
│   ├── 2b_Target_Encoding_Preprocess.py           # processes raw data for modeling
│   ├── 3_main.py                                  # primary script for training and comparing model performance
│   ├── README.md  
│   ├── setup.py  
│  
├── data/  
│   ├── BOA_database_for_exercise_from_2020.json  #raw data
│   ├── processed/                                # Processed data output for main modeling script
│  
├── metrics/                                      # Metric outputs to compare model performance
├── models/                                       # Saved models 
├── output/                                       # Export location for key findings
├── reference_materials/                          # All original source materials for the projectg
└── src/                                          # source code folder for all stored functions and metadata
    ├── config.yaml  
    ├── config_loader.py  
    ├── __init__.py  
    ├── epo_utils/  
    ├── your_project.egg-info/    
    └── __pycache__/  
        ├── config_loader.cpython-311.pyc  
        ├── __init__.cpython-310.pyc  
        ├── __init__.cpython-311.pyc  
    |   |   epo_search.py        # Search utilities for EPO data
    |   |   feature_selection.py # Feature selection logic
    |   |   preprocessing.py     # Data preprocessing utilities
    |   |   summary_stats.py     # Generates summary statistics
    |   |   __init__.py
    |
    +---your_project.egg-info   # Package metadata
    |
    \---__pycache__             # Compiled Python files
```

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- Required dependencies (specified in `setup.py` or a `requirements.txt` file)

### Steps
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install dependencies:
   ```sh
   pip install -e .
   ```
3. (Optional) Set up a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

## Usage
- **Run EDA**: `python EDA.py`
- **Run preprocessing**: `python preprocessing.py`
- **Execute main pipeline**: `python main.py`

## Configuration
Settings can be modified in `config.yaml` to adjust paths for reading and writing.

## Contact
For questions or issues, please reach out via GitHub issues.

