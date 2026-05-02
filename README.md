# Customer Churn Prediction System (MLOps Project)

## Project Overview

This project develops a machine learning model to predict customer churn using historical customer data. The goal is to identify customers at high risk of leaving so that targeted retention strategies can be applied.

The final solution includes:
- A trained Gradient Boosting model
- Experiment tracking using MLflow
- A deployed Flask API for real-time predictions
- A defined decision threshold for business use

---

## Dataset Description

The dataset includes customer demographic information and behavioral features such as:
- Account tenure
- Service usage patterns
- Contract type
- Billing information
- Customer support interactions

The target variable is:
- **Churn (1 = customer left, 0 = retained)**

---

## Methodology

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling where applicable
- Train-test split for evaluation

### 2. Feature Engineering
- Transformation of categorical variables
- Creation of derived behavioral indicators (if applicable)

### 3. Models Tested
The following models were evaluated using MLflow tracking:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier (final model)
- Tuned Gradient Boosting

---

## Final Model Selection

The final model selected was: **Gradient Boosting Classifier**

### Selection Criteria:
- Highest ROC-AUC score
- Strong balance between precision and recall
- Best performance on minority (churn) class

---

## Evaluation Metrics

The following metrics were used to evaluate all models:

- Accuracy
- Precision (Churn class)
- Recall (Churn class)
- F1-Score
- ROC-AUC

---

## Decision Threshold

Instead of using the default 0.50 threshold, a custom business threshold was applied: > **Threshold = 0.35**

### Why 0.35?
- Improves recall for churn detection
- Prioritizes identifying at-risk customers
- Supports retention-focused business strategy

### Output Labels:
- Probability ≥ 0.35 → **High Risk (Churn)**
- Probability < 0.35 → **Low Risk (Retained)**

---

## Flask API Deployment

The model is deployed using a Flask API.

### Available Endpoints:

#### 1. Health Check

  Endpoint:
  
    GET /health

  Response:

    {
      "status": "ok"
    }

#### 2. Single Prediction

  Endpoint:

    POST /predict

  Example Request:

    {
      "tenure": 5,
      "MonthlyCharges": 85.5,
      "TotalCharges": 300,
      "Contract": "Month-to-month",
      "PhoneService": "Yes",
      "StreamingTV": "Yes"
    }
  
  Example Response:

    {
      "prediction": 1,
      "churn_probability": 0.7421,
      "risk_label": "High Risk"
    }
    
#### 3. Batch Prediction

  Endpoint:

    POST /predict/batch

  Example Request:

    [
      {
        "tenure": 5,
        "MonthlyCharges": 85.5,
        "TotalCharges": 300
      },
      {
        "tenure": 24,
        "MonthlyCharges": 60,
        "TotalCharges": 1500
      }
    ]

  Example Response:

    {
      "results": [
        {
          "prediction": 1,
          "churn_probability": 0.7421,
          "risk_label": "High Risk"
        },
        {
          "prediction": 0,
          "churn_probability": 0.1832,
          "risk_label": "Low Risk"
        }
      ]
    }

## Error Handling:

  Missing Field Example:

    {
      "error": "Missing required field: tenure"
    }

  Invalid Input Example:

    {
      "error": "Invalid input: contains non-numeric values"
    }

## API Testing

Automated tests are included in: test_api.py

Run tests: python test_api.py

Tests include:

- Health check validation
- Single prediction
- Batch prediction
- Missing field handling
- Invalid input handling

---

## Setup Instructions

1. Clone Repository:

   git clone https://github.com/chris-chabrier/DATA-6545-Final-Project-Telco-Churn.git
   cd DATA-6545-Final-Project-Telco-Churn

2. Create Virtual Environment:

   python -m venv venv
   source venv/bin/activate

3. Install Dependencies:

   pip install -r requirements.txt

4. Run API:

   python app.py

   (API runs at: http://127.0.0.1:5000)

5. Run Tests:

   python test_api.py
