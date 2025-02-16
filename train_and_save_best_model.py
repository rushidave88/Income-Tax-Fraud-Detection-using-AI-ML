import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
from xgboost import XGBClassifier

# Load your dataset
data = pd.read_csv("C:/Users/rushi/Desktop/income_regression_dataset.csv")

# Preprocess your data
label_encoder_occupation = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_children = LabelEncoder()

data['Occupation'] = label_encoder_occupation.fit_transform(data['Occupation'])
data['Marital_Status'] = label_encoder_marital.fit_transform(data['Marital_Status'])
data['Children (Yes/No)'] = label_encoder_children.fit_transform(data['Children (Yes/No)'])

# Save the label encoders
joblib.dump(label_encoder_occupation, "label_encoder_occupation.joblib")
joblib.dump(label_encoder_marital, "label_encoder_marital_status.joblib")
joblib.dump(label_encoder_children, "label_encoder_children.joblib")

# Drop any non-numeric columns that should not be included in the model training
data = data.drop(columns=['Name', 'PAN_Card', 'Aadhar_Card', 'Bank_Account_No'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the data into training and testing sets
X = data_imputed.drop(columns=["Reported_Income"])
y = data_imputed["Reported_Income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models and find the best one
models = {
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=50),  # Reduced number of trees
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(n_estimators=50)  # Reduced number of boosting rounds
}

model_accuracies = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy
    except MemoryError as e:
        print(f"MemoryError with model {name}: {e}")

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]

# Save the best model
joblib.dump(best_model, "best_model.joblib")
print(f"The best performing model is {best_model_name} and has been saved as 'best_model.joblib'")