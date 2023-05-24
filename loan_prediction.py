import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Read the dataset
loan_df = pd.read_csv("loan.csv")
loan_df = loan_df[['gender','married','dependents','self employed','education','loan status']]

# Encode categorical variables
number = LabelEncoder()
loan_df['gender'] = number.fit_transform(loan_df['gender'])
loan_df['married'] = number.fit_transform(loan_df['married'])
loan_df['dependents'] = number.fit_transform(loan_df['dependents'])
loan_df['self employed'] = number.fit_transform(loan_df['self employed'])
loan_df['education'] = number.fit_transform(loan_df['education'])
loan_df['loan status'] = number.fit_transform(loan_df['loan status'])

# Define features and target
features = ["gender", "married", "dependents", 'self employed', 'education']
target = "loan status"

# Split the dataset into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(loan_df[features], loan_df[target], test_size=0.50, random_state=42)

# Creating the Model
model = GaussianNB()

# Fit the model on the training data
model.fit(features_train, target_train)

# Make predictions on the testing data
predictor = model.predict(features_test)
accuracy = accuracy_score(target_test, predictor)
print("\nModel Accuracy =", accuracy * 100, "%")

# Make predictions on new data
new_data = [[1, 0, 1, 1, 1]]
new_data_df = pd.DataFrame(new_data, columns=features)
prediction = model.predict(new_data_df)

if prediction[0] == 1:
    print("\nApproved")
elif prediction[0] == 0:
    print("\nNot Approved")

joblib.dump(model, "loan_model.joblib")
