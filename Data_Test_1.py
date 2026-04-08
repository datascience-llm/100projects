import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
path = "/content/sample_data/california_housing_test.csv"
df = pd.read_csv(path)

# Basic check
print(df.head())

# Plot histogram of population
plt.figure(figsize=(8, 5))
plt.hist(df["population"], bins=30, color="skyblue", edgecolor="black")

# Labels and title
plt.xlabel("Population")
plt.ylabel("Frequency")
plt.title("Distribution of Population in California Housing Dataset")

# Grid for better readability
plt.grid(axis="y", alpha=0.75)

plt.show()
