import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2

print("\nThis is the test with Feature Selection\n")

# Load model from file
with open('model_chi.pkl', 'rb') as file:
    model = pickle.load(file)


def factorize_string_columns(df):
    for column in df.select_dtypes(include=['object']):
        df[column] = pd.factorize(df[column])[0]
    return df

# Load NSL-KDD dataset
data = pd.read_csv('NSL-KDD/KDDTest+.txt', header=None)

# Convert categorical columns to numerical data
data = factorize_string_columns(data)

# split the data into features and target variable
X = pd.concat([data.iloc[:, :40], data.iloc[:, 42:]], axis=1)
y = data.iloc[:, -2]


# apply feature selection
selector = SelectKBest(chi2, k=31)
X_new = selector.fit_transform(X, y)

# get the selected feature indices
selected_features = selector.get_support(indices=True)

# print the selected feature names
feature_names = list(X.columns)
selected_feature_names = [feature_names[i] for i in selected_features]

selected_features = data.iloc[:, selected_feature_names]
X = selected_features.iloc[:, :]


#Preprocessing 
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Use model to make predictions
y_pred = model.predict(X)

# Evaluate model
acc = accuracy_score(y, y_pred)
print('Accuracy:', acc, '\n')

file.close()