import pandas as pd

# Load dataset
df = pd.read_csv("student_scores_clean.csv")

# Display original dataset
print("Original Dataset")
print(df)

# Check missing values
print("\nMissing Values")
print(df.isnull())

print("\nMissing Values Count")
print(df.isnull().sum())

# Fill missing values
df = df.fillna(0)

print("\nDataset After Filling Missing Values")
print(df)

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset After Removing Duplicates")
print(df)

# Dataset statistics
print("\nDataset Statistics")
print(df.describe())