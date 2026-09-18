================================================================================
CUSTOMER CHURN PREDICTION CLI TOOL - README
================================================================================

An end-to-end Machine Learning Command Line Interface (CLI) application 
built using Python, Scikit-Learn, and Click. This tool allows users to train 
a Machine Learning model on customer churn datasets, evaluate performance 
metrics, and generate churn predictions directly from the terminal.

--------------------------------------------------------------------------------
TABLE OF CONTENTS
--------------------------------------------------------------------------------
1. Project Overview
2. Key Features
3. Repository Structure
4. Requirements
5. Installation Guide
6. Usage Instructions
7. Dataset Details
8. License

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
Customer Churn is one of the most critical metrics for subscription-based 
business models. This CLI application provides a lightweight, terminal-executable 
solution to predict whether a customer will churn based on historical usage, 
account tenure, payment methods, and demographic data.

The project is structured strictly for command-line execution without requiring 
any graphical user interface (GUI).

--------------------------------------------------------------------------------
2. KEY FEATURES
--------------------------------------------------------------------------------
- Pure CLI Workflow: Completely executable from the command line using flags and arguments.
- Automated Data Preprocessing: Handles missing values, performs feature scaling, and encodes categorical variables automatically.
- Model Training & Evaluation: Trains a Random Forest Classifier and displays detailed performance evaluation metrics (Precision, Recall, F1-Score, ROC-AUC).
- Batch & Single Prediction: Supports predictions on new batch CSV datasets as well as custom command-line parameter inputs.
- Model Persistence: Automatically exports and reloads trained model artifacts (.pkl).

--------------------------------------------------------------------------------
3. REPOSITORY STRUCTURE
--------------------------------------------------------------------------------
churn-prediction-cli/
├── data/
│   ├── churn_data.csv            # Sample dataset for training
│   └── new_customers.csv         # Sample dataset for prediction
├── models/
│   └── churn_model.pkl           # Saved trained model pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data ingestion and cleaning
│   ├── preprocess.py             # Feature encoding and scaling
│   ├── train.py                  # Model training logic
│   └── predict.py                # Inference pipeline
├── main.py                       # CLI application entry point
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation

--------------------------------------------------------------------------------
4. REQUIREMENTS
--------------------------------------------------------------------------------
- Python 3.8 or higher
- pip package manager

Main Libraries Used:
- pandas (Data manipulation)
- numpy (Numerical computations)
- scikit-learn (Machine Learning pipeline and evaluation)
- joblib (Model serialization)
- argparse / click (Command Line Interface framework)

--------------------------------------------------------------------------------
5. INSTALLATION GUIDE
--------------------------------------------------------------------------------
Step 1: Clone the Repository
  git clone https://github.com/your-username/churn-prediction-cli.git
  cd churn-prediction-cli

Step 2: Create a Virtual Environment (Optional but Recommended)
  # On Linux/macOS
  python3 -m venv venv
  source venv/bin/activate

  # On Windows
  python -m venv venv
  venv\Scripts\activate

Step 3: Install Dependencies
  pip install -r requirements.txt

--------------------------------------------------------------------------------
6. USAGE INSTRUCTIONS
--------------------------------------------------------------------------------
The project is controlled via main.py using command-line arguments.

1. Train the Model:
  python main.py train --data-path data/churn_data.csv

  Expected Output:
  [INFO] Loading dataset from data/churn_data.csv...
  [INFO] Preprocessing data and splitting train/test sets...
  [INFO] Training Random Forest Classifier...
  ==================================================
  Model Evaluation Metrics:
  --------------------------------------------------
  Accuracy  : 85.40%
  Precision : 81.20%
  Recall    : 78.50%
  F1 Score  : 79.82%
  ROC-AUC   : 0.887
  ==================================================
  [INFO] Model saved successfully to models/churn_model.pkl

2. Run Predictions on New Data:
  python main.py predict --input-path data/new_customers.csv --output-path data/predictions.csv

  Expected Output:
  [INFO] Loading model from models/churn_model.pkl...
  [INFO] Processing input batch file...
  [INFO] Predictions completed!
  [INFO] Results exported to data/predictions.csv

--------------------------------------------------------------------------------
7. DATASET DETAILS
--------------------------------------------------------------------------------
The model expects a CSV file containing standard customer attributes:
- Tenure: Number of months the customer has stayed with the company.
- MonthlyCharges: The amount charged to the customer monthly.
- TotalCharges: The total amount charged to the customer.
- Contract: Contract type (Month-to-month, One year, Two year).
- PaymentMethod: Payment method used by customer.
- Churn: Target variable (1 for Churned, 0 for Retained).

--------------------------------------------------------------------------------
8. LICENSE
--------------------------------------------------------------------------------
This project is open-source under the MIT License.
