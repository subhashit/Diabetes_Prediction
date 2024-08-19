import pandas as pd
import numpy as np
from collections import Counter
import math

# Loading and data processing
df = pd.read_csv(r'C:\Users\subha\Desktop\SUBHASHIT THAPA\project\KNN_fromscratch\diabetes (4) (1).csv')

non_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction']
for col in non_zero:
    df[col] = df[col].replace(0, np.nan)
    mean = np.nanmean(df[col])
    df[col] = df[col].replace(np.nan, mean)

# splitting the data
df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
test_size = 0.2
split_index = int(len(df_shuffled) * (1 - test_size))

x_train = df_shuffled[:split_index]
x_test = df_shuffled[split_index:]

# labeling 
train_data = {0: [], 1: []}
test_data = {0: [], 1: []}
for i in range(len(x_train)):
    train_data[x_train.iloc[i, -1]].append(x_train.iloc[i, :-1].values)

for i in range(len(x_test)):
    test_data[x_test.iloc[i, -1]].append(x_test.iloc[i, :-1].values)

# Implementing KNN
k = int(math.sqrt(len(x_train)))

def knn(data, predict, k):
    distance = []
    for group in data:
        for features in data[group]:
            dist = np.linalg.norm(np.array(features) - np.array(predict))
            distance.append([dist, group])
    votes = [i[1] for i in sorted(distance)[:k]]
    result = Counter(votes).most_common(1)[0][0]
    return result

# Test the model
correct = 0
total = 0
for group in test_data:
    for data in test_data[group]:
        vote = knn(train_data, data, k)
        if vote == group:
            correct += 1
        total += 1

print(f"Accuracy of model build from scratch: {correct / total}")

#KNN model from scikit learn
from sklearn.model_selection import train_test_split
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier  = KNeighborsClassifier(n_neighbors=11, p=2, metric="euclidean")
classifier.fit(x_train,y_train)

predict=classifier.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predict)
print("accuracy_score of skleran's KNeighborsClassifier model = ", accuracy)

