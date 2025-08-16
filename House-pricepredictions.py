# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# Creating Datasets
X = np.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250]).reshape(-1, 1)  # Size in sqft
Y = np.array([15, 22, 29, 36, 43, 50, 57, 64])  # Price in Lakhs
# Initialize and train the Model
model = LinearRegression()
model.fit(X, Y)
# Predict on training Data
Y_pred = model.predict(X)
#Visualize the Data
plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Predicted Line')
plt.xlabel('Size (sqft)')
plt.ylabel('Price (Lakhs)')
plt.title('Linear Regression Model : Size vs Price')
plt.legend()
plt.grid(True)
plt.show()
# Evaluation of the model
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Y_pred)
print("model Evaluation:")
print(f" RMSE : {rmse:.3f}")
print(f" R2 Score : {r2:.3f}")
# Printing model co-efficient
print("learned parameters \n")
print(f"slope(w) : {model.coef_[0]:.3f}")
print(f"intercept(b) : {model.intercept_:.3f}")
# Printing model co-efficient
print("learned parameters \n")
print(f"slope(w) : {model.coef_[0]:.3f}")
print(f"intercept(b) : {model.intercept_:.3f}")
# Testing Model
X_test = np.array([[1000], [1600], [2300]])  # Test inputs: sizes in sqft
Y_test_pred = model.predict(X_test)          # Predict prices using the trained model

# Display the predictions
for x_val, y_val in zip(X_test.flatten(), Y_test_pred):
    print(f"House size: {x_val} sqft --> Predicted Price: ₹{y_val:.2f} Lakhs")
