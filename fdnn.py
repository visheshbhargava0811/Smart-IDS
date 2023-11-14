import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

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
print("Dataset after with fs:: ",X.shape)


#Preprocessing 
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=None)



# Define the model architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(patience=5)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('\n\nTest accuracy with fs::', accuracy, "\n\n")