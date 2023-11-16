import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets and create two dataframes
train_data = pd.read_csv("./titanic/train.csv")
test_data = pd.read_csv("./titanic/test.csv")

# Find out what values are missing 
# print(train_data.isna().sum()) 
# print(test_data.isna().sum())

# Fill missing values with mean imputation 
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(train_data.mean(), inplace=True)
# print(train_data.isna().sum()) 
# print(test_data.isna().sum())

# Dropping unnecessary features 
train_data = train_data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)
test_data = test_data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)

train_data.info()

# We use label encoding to convert Sex to numerical
labelEncoder = LabelEncoder()
labelEncoder.fit(train_data["Sex"])
labelEncoder.fit(test_data["Sex"])
train_data["Sex"] = labelEncoder.transform(train_data["Sex"])
test_data["Sex"] = labelEncoder.transform(test_data["Sex"])

# Creating arrays of train_data and dropping survived
X = np.array(train_data.drop(['Survived'], 1).astype(float))
Y = np.array(train_data['Survived'])

# Scale the values of the features to same range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X); 

kmeans = kmeans = KMeans(n_clusters=2); 
kmeans.fit(X_scaled); 
KMeans(algorithm="lloyd", copy_x=True, init="k-means++", max_iter=100000, n_clusters=2, n_init=10000, random_state=None, tol=0.0001, verbose=0)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))