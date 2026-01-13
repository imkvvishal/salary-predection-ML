# salary_prediction_input.py

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load CSV file
data = pd.read_csv("salary_data.csv")  # <-- Replace with your CSV file name
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Explore the data
print("\nData info:")
print(data.info())
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop missing values (if any)
data = data.dropna()

# Step 4: Select features and target
X = data[['Experience']]  # Feature column(s)
y = data['Salary']             # Target column

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions for test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 9: Visualize the results
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_test.iloc[:,0], y=y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.legend()
plt.show()

# Step 10: Take user input for experience
while True:
    try:
        user_exp = float(input("\nEnter years of experience to predict salary (or type -1 to exit): "))
        if user_exp == -1:
            print("Exiting...")
            break
        predicted_salary = model.predict(np.array([[user_exp]]))
        print(f"Predicted Salary for {user_exp} years of experience: {predicted_salary[0]:.2f}")
    except ValueError:
        print("Please enter a valid number!")
