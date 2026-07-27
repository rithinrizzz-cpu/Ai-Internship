import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("student_marks.csv")

# Scatter Plot
plt.scatter(df["Math"], df["Science"])
plt.title("Math vs Science Marks")
plt.xlabel("Math Marks")
plt.ylabel("Science Marks")
plt.show()

# Bar Chart
plt.bar(df["Name"], df["Math"])
plt.title("Math Marks of Students")
plt.xlabel("Student Name")
plt.ylabel("Math Marks")
plt.show()

# Line Chart
plt.plot(df["Name"], df["English"], marker="o")
plt.title("English Marks")
plt.xlabel("Student Name")
plt.ylabel("English Marks")
plt.show()