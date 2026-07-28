import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_marks7.csv")

print("Dataset")
print(df)

# Input and Output
X = df[["Hours"]]
y = df["Marks"]

# Split data
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

# Predict
prediction = model.predict([[5]])

print("Predicted Marks for 5 Hours:")
print(prediction)