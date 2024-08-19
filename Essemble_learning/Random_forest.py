import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r'C:\Users\subha\Desktop\SUBHASHIT THAPA\project\KNN_fromscratch\diabetes (4) (1).csv')

# Replace zeros with NaN and then fill with column mean
non_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction']
for col in non_zero:
    df[col] = df[col].replace(0, np.nan)
    mean = np.nanmean(df[col])
    df[col] = df[col].replace(np.nan, mean)

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Initialize Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=0
)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy for Random Forest: {accuracy :.4f}')

