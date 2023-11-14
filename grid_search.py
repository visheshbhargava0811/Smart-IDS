import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

def factorize_string_columns(df):
    for column in df.select_dtypes(include=['object']):
        df[column] = pd.factorize(df[column])[0]
    return df

# Load KDDTrain+ dataset
data = pd.read_csv('NSL-KDD/KDDTrain+.txt', header=None)

# Convert categorical columns to numerical data
data = factorize_string_columns(data)

# Preprocessing
X = pd.concat([data.iloc[:, :40], data.iloc[:, 42:]], axis=1)
y = data.iloc[:, -2] # labels

# apply feature selection
selector = SelectKBest(chi2, k=21)
X_new = selector.fit_transform(X, y)

# get the selected feature indices
selected_features = selector.get_support(indices=True)

# print the selected feature names
feature_names = list(X.columns)
selected_feature_names = [feature_names[i] for i in selected_features]

selected_features = data.iloc[:, selected_feature_names]
X = selected_features.iloc[:, :]

encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# set the parameter grid
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# create an SVM model
svm = SVC()

# perform grid search using cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5)

# fit the model to the training data
grid_search.fit(X_train, y_train)

# predict the labels of the test data using the best model
y_pred = grid_search.predict(X_test)

# calculate the accuracy of the model on the test data
accuracy = accuracy_score(y_test, y_pred)

# print the best hyperparameters and the accuracy of the model
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy: ", accuracy)

