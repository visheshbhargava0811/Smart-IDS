import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.feature_selection import SelectKBest, chi2

print("\nWith Feature Selection\n")

def factorize_string_columns(df):
    for column in df.select_dtypes(include=['object']):
        df[column] = pd.factorize(df[column])[0]
    return df

# Load NSL-KDD dataset
data = pd.read_csv('NSL-KDD/KDDTrain+.txt', header=None)

# Convert categorical columns to numerical data
data = factorize_string_columns(data)
print("Dataset before:: ",data.shape)


# split the data into features and target variable
X = pd.concat([data.iloc[:, :40], data.iloc[:, 42:]], axis=1)
y = data.iloc[:, -2] #labels

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
print("Dataset after:: ",X.shape)


#Preprocessing 
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=12)


# Train model
model = RandomForestClassifier(n_estimators=50, criterion='gini' ,random_state=123)
model.fit(X, y)


# Test model
y_pred = model.predict(X_test)

# Evaluate model
acc = accuracy_score(y_test, y_pred)
print('\nAccuracy:', acc)


# Save model to file
with open('model_chi.pkl', 'wb') as file:
    pickle.dump(model, file)

file.close()
print("Model saved succesfully\n")