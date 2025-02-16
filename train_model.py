import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv("C:/Users/rushi/Desktop/income_regression_dataset.csv")

# Encode categorical variables
le_occupation = LabelEncoder()
le_marital = LabelEncoder()
le_children = LabelEncoder()

df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
df['Marital_Status'] = le_marital.fit_transform(df['Marital_Status'])
df['Children (Yes/No)'] = le_children.fit_transform(df['Children (Yes/No)'])

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Features and target
X = df.drop(columns=['Reported_Income'])
y = df['Reported_Income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save the encoders and model
joblib.dump(le_occupation, 'label_encoder_occupation.joblib')
joblib.dump(le_marital, 'label_encoder_marital_status.joblib')
joblib.dump(le_children, 'label_encoder_children.joblib')
joblib.dump(model, 'best_model.joblib')

print("Model and encoders saved successfully.")
