import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_marks9.csv")

print("Dataset")
print(df)

# Input and Output
X = df[["Hours"]]
y = df["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predictions
study_hours = [[2], [4], [6], [8]]

predictions = model.predict(study_hours)

print("\nPredicted Student Scores")

for hour, mark in zip(study_hours, predictions):
    print(f"Study Hours: {hour[0]} -> Predicted Marks: {mark:.2f}")