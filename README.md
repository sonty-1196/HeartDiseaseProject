# HeartDiseaseProject
Heart Disease Prediction 
Heart Disease Prediction using Logistic Regression
Overview
This project focuses on predicting the 10-year risk of Coronary Heart Disease (CHD) using the Framingham Heart Study dataset. The objective is to build a machine learning classification model that can assist in early identification of cardiovascular risk.

The model is developed using Logistic Regression and evaluated using multiple performance metrics to ensure reliability and interpretability.

Problem Statement
Cardiovascular disease is one of the leading causes of mortality worldwide. Early risk assessment can significantly reduce severe outcomes through preventive measures and timely intervention.
This project aims to:

Predict whether a patient will develop CHD within 10 years

Analyze clinical and lifestyle risk factors

Evaluate classification performance using statistical metrics

Interpret model coefficients to understand feature impact

Dataset Information

The dataset used in this project is the Framingham Heart Study dataset, which contains demographic, behavioral, and medical attributes of patients.

Target Variable:
TenYearCHD

0 → No 10-year CHD risk

1 → 10-year CHD risk present

Key Features Include:
Age
Sex
Smoking Status
Diabetes
Total Cholesterol
Systolic Blood Pressure
Diastolic Blood Pressure
Body Mass Index (BMI)
Heart Rate
Glucose

Technologies Used are:
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

Machine Learning Approach
Data Loading and Cleaning

Exploratory Data Analysis

Feature Scaling using StandardScaler

Train-Test Split (80-20)

Logistic Regression Model Training

Model Evaluation using:

Accuracy

Confusion Matrix

ROC Curve

AUC Score

Feature Importance Analysis

Model Performance

Accuracy: 85%

ROC-AUC Score: (Add your value here)

Evaluation Metrics: Precision, Recall, F1-Score

The model demonstrates strong predictive capability in identifying individuals at risk of heart disease.

Visualizations Included

Target Variable Distribution

Correlation Heatmap

Confusion Matrix

ROC Curve

Feature Importance Plot

Project Structure
Heart-Disease-Prediction/

Heart_Disease_prediction_project.ipynb
framingham.csv
README.md
requirements.txt

How to Run the Project
Clone the repository:
git clone https://github.com/your-username/heart-disease-prediction.git

Install dependencies:
pip install -r requirements.txt

Run the Jupyter Notebook:
jupyter notebook

Business / Practical Impact
This model can assist healthcare professionals in:

Early risk detection

Preventive screening

lowering down long-term cardiovascular difficulties.

Supporting clinical decision-making

It highlights and shows how machine learning can be used in real-world healthcare analytics.

Future Improvements

Hyperparameter tuning

Cross-validation

Handling class imbalance (SMOTE)

Comparing with advanced models (Random Forest, XGBoost)

Deployment as a web application

Author
Daewansh Bhagwat Prasad Bansal
Internship Project – Heart Disease Prediction
