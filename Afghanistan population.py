print("Program started")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load CSV with correct encoding
data = pd.read_csv("population_afghanistan.csv", encoding="latin1")

print("CSV loaded successfully")
print(data.head())

# =====================
# DATA CLEANING
# =====================
data["Population"] = data["Population"].str.replace(",", "").astype(float)
data["Year"] = pd.to_numeric(data["Year"])

data = data.dropna()
data = data.sort_values(by="Year")

# =====================
# LINE GRAPH
# =====================
plt.figure()
plt.plot(data["Year"], data["Population"])
plt.title("Population Growth of Afghanistan")
plt.xlabel("Year")
plt.ylabel("Population")
plt.show()

# =====================
# SCATTER PLOT
# =====================
plt.figure()
plt.scatter(data["Year"], data["Population"])
plt.title("Population Distribution")
plt.xlabel("Year")
plt.ylabel("Population")
plt.show()

# =====================
# MACHINE LEARNING
# =====================
X = data[["Year"]]
y = data["Population"]

model = LinearRegression()
model.fit(X, y)

future_years = np.array([[2030], [2040], [2050]])
predictions = model.predict(future_years)

print("Population Predictions:")
for year, pop in zip(future_years, predictions):
    print(f"Year {year[0]} -> Population: {int(pop)}")

# =====================
# REGRESSION GRAPH
# =====================
plt.figure()
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("Linear Regression Model")
plt.xlabel("Year")
plt.ylabel("Population")
plt.show()

print("Program finished successfully")
input("Press Enter to exit")
