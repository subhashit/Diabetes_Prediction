import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\subha\Desktop\SUBHASHIT THAPA\project\KNN_fromscratch\diabetes (4) (1).csv')

non_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction']
for col in non_zero:
    df[col] = df[col].replace(0, np.nan)
    mean = np.nanmean(df[col])
    df[col] = df[col].replace(np.nan, mean)

# Define features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# Base models
estimators = [
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True))
]
# Meta-model
meta_model = LogisticRegression()
# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model
)
stacking_clf.fit(X_train, y_train)

y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy for stacking: {accuracy:.4f}')
