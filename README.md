# Diabetes Prediction and Model Comparison

## Overview:
This project aims to predict whether a person has diabetes based on specific health indicators using the famous Diabetes dataset. The project involves building and comparing multiple machine learning models to evaluate their performance in predicting diabetes. The models implemented include:

1. K-Nearest Neighbors (KNN) implemented from scratch

2. K-Nearest Neighbors (KNN) using scikit-learn

3. Stacking Classifier using scikit-learn

4. Random Forest using scikit-learn

## Dataset:
 The dataset used in this project is the Pima Indians Diabetes Database, which is available in the public domain. It contains 768 observations with 8 features and a binary outcome:

1. Pregnancies: Number of times pregnant

2. Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance 

3. BloodPressure: Diastolic blood pressure (mm Hg)
  
4. SkinThickness: Triceps skinfold thickness (mm)
   
5. Insulin: 2-Hour serum insulin (mu U/ml)

6. BMI: Body mass index (weight in kg/(height in m)^2)

7. DiabetesPedigreeFunction: Diabetes pedigree function

8. Age: Age in years

9. Outcome: Binary outcome, 1 indicates diabetes, 0 indicates no diabetes

## Results:

KNN (from scratch): 0.7207 accuracy

KNN (scikit-learn): 0.8311 accuracy

Stacking (scikit-learn): 0.8442 accuracy

Random Forest (scikit-learn): 0.8182 accuracy

## Conclusion:
This project demonstrates how different machine learning algorithms can be applied to a medical dataset to predict health outcomes. By comparing the accuracy of different models, we can identify the most effective approach for this specific problem.

