# Heart Disease Risk Prediction Model

##  Overview

This project focuses on building a **supervised machine learning classification model** to predict the likelihood of heart disease based on patient health data.
The objective is to assist in **early risk detection** by analyzing key medical features using data-driven techniques.

##  Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early detection can significantly improve patient outcomes.
This project uses machine learning algorithms to classify whether a patient is at risk of heart disease based on clinical attributes.

##  Tech Stack & Tools

1. Programming Language: Python
2. Libraries: NumPy, Pandas, Scikit-learn, Matplotlib
3. Algorithms: Decision Tree, Random Forest
4. Techniques:

  * Data Cleaning & Preprocessing
  * Exploratory Data Analysis (EDA)
  * Supervised Learning
  * Model Evaluation

##  Dataset

The dataset contains patient health-related features such as:

* Age
* Sex
* Blood pressure
* Cholesterol levels
* Heart rate
* Other relevant medical indicators

##  Data Preprocessing

* Handled missing and inconsistent values
* Converted categorical features into numerical format
* Scaled and formatted features for model compatibility
* Split data into **training and testing sets

##  Exploratory Data Analysis (EDA)

* Analyzed feature distributions and correlations
* Visualized relationships between health indicators and heart disease risk
* Used **Matplotlib** to identify patterns and trends in the data

##  Model Training

The following supervised learning models were trained using Scikit-learn:

1. Decision Tree Classifier
2. Random Forest Classifier

##  Model Evaluation

Models were evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall

The Random Forest model demonstrated better generalization and reduced overfitting, making it the preferred model for prediction.

##  Results

* Achieved high prediction accuracy on test data
* Random Forest outperformed Decision Tree in terms of overall reliability
* Model effectively identifies patients at risk of heart disease

##  Future Enhancements

* Add real-time data input support
* Deploy model using a web interface
* Improve generalization using larger and more diverse datasets
* Add cross-validation and hyperparameter tuning
* 
