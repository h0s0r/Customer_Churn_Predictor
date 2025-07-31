# Predictive Modeling: Telco Customer Churn Analysis

## Project Overview

This project focuses on building and evaluating a machine learning model to predict customer churn for a telecommunications company. The goal is to identify customers who are likely to cancel their service, enabling the business to proactively target them with retention offers. This repository contains the complete workflow, from raw data cleaning to model evaluation and visualization.

## Dataset

The project utilizes the "WA_Fn-UseC_-Telco-Customer-Churn" dataset, a popular real-world dataset sourced from Kaggle. It contains customer account information, demographic data, and details on the services they subscribe to.

## Project Workflow

The analysis and modeling process was executed in the following sequence:

1.  **Data Cleaning & Preprocessing:**
    * Loaded the raw dataset and performed an initial exploration to identify data quality issues.
    * Conducted advanced data cleaning by identifying and correcting the `TotalCharges` column, which was incorrectly typed as an `object` due to hidden empty string values. These were handled and the column was converted to a `float`.

2.  **Feature Encoding:**
    * The binary target variable `Churn` was converted from 'Yes'/'No' strings to a numerical `1`/`0` format.
    * **One-Hot Encoding** was applied to all 18 categorical feature columns to transform them into a numerical format suitable for the machine learning model.

3.  **Data Splitting & Scaling:**
    * The dataset was separated into features (`X`) and the target (`y`).
    * The data was then split into an 80% training set and a 20% testing set.
    * **Feature Scaling** was applied to all features using `StandardScaler` to normalize their ranges and ensure they contribute equally to the model's performance.

4.  **Model Training:**
    * A **`LogisticRegression`** model was trained on the fully preprocessed training data.

5.  **Model Evaluation & Visualization:**
    * The trained model's performance was evaluated on the unseen test data.
    * Key classification metrics including **Precision, Recall, F1-Score,** and the **Confusion Matrix** were calculated.
    * A **heatmap** was generated using Seaborn to provide a clear visual representation of the confusion matrix.

## Skills & Technologies

* **Python**
* **Pandas:** For data manipulation, cleaning, and encoding.
* **Scikit-learn:** For data splitting, scaling, model training, and evaluation.
* **Matplotlib & Seaborn:** For data visualization.
* **Data Cleaning & Preprocessing**
* **Feature Scaling**
* **Logistic Regression**
* **Classification Model Evaluation**