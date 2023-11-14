import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

print("\nSMV without Feature Selction\n")

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
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train SVM model
model = SVC(kernel='linear', C=25, random_state=231)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy score
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)

# Calculate precision, recall, and F1 scores
precision = precision_score(y_test, y_pred, average='macro', zero_division=True)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1, "\n")

'''
# Plot the accuracy score over time
plt.plot(model.decision_function(X_test), label='Decision Function')
plt.plot(y_test, label='True Label')
plt.legend()
plt.show() '''