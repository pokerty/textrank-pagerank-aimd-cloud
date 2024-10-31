import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ITERATESMAX = 100  # You can adjust this value as needed 
C = 10  # Adjust C value as needed
exponent1 = 0.5  # Adjust exponent1 value as needed (should be less than 1 for decrease)
exponent2 = 0.5  # Adjust exponent2 value as needed (should be less than 1 for decrease)

ITERATESMAX = 100 # You can adjust this value as needed 
C = 10 # Adjust C value as needed
# Experiment 4
# C = 3
alpha = 0.1 # Adjust alpha value as needed
exponent1 = 0.5 # Adjust exponent1 value as needed (beta)
exponent2 = 0.5 # Adjust exponent2 value as needed (beta)
# Experiment 2
# exponent2 = 0.6

x1 = 1
x2 = 1 
alpha1 = 1
alpha2 = 1
# Experiment 2 
# alpha2 = 1.2

x1_values = np.zeros(ITERATESMAX) 
x2_values = np.zeros(ITERATESMAX)
resource_allocation = np.zeros(ITERATESMAX)

# Generate sample data for training the machine learning model
num_samples = 1000
resource_utilization_history = np.random.rand(num_samples, 1)  # Example: resource utilization history (normalized)

# Predictive horizon for resource utilization
predictive_horizon = 10  

# Create a history matrix with columns representing past resource utilizations and rows representing different time steps
history_matrix = np.zeros((num_samples - predictive_horizon, predictive_horizon))
for i in range(predictive_horizon):
    history_matrix[:, i] = resource_utilization_history[i:num_samples - predictive_horizon + i].flatten()

# Target values for alpha (for training the predictive model)
target_alpha = resource_utilization_history[predictive_horizon:]

# Train linear regression model
model = LinearRegression()
model.fit(history_matrix, target_alpha)

for i in range(ITERATESMAX):
    # Use the trained machine learning model to predict future resource utilization
    future_resource_utilization = np.zeros((1, predictive_horizon))
    future_resource_utilization[0] = resource_utilization_history[-predictive_horizon:].flatten()
    predicted_resource_utilization = model.predict(future_resource_utilization)
    predicted_alpha = max(0, min(1, predicted_resource_utilization[-1]))  # Ensure predicted alpha is in the range [0, 1]
    if (x1 + x2 <= C):
        # Additive increase phase 
        print('Additive I')
        # Experiment 1 
        # alpha1 = alpha * np.power(x1, exponent1) 
        # alpha2 = alpha * np.log(x2 + 1)
        # Experiment 3
        # alpha1 = alpha * x1
        # alpha2 = alpha / x2
        # Experiment 5
        alpha1 = alpha * np.power(x1, exponent1) 
        alpha2 = predicted_alpha

        x1 = x1 + alpha1
        x2 = x2 + alpha2 
    else:
        # Simulate network condition (for example, congestion) 
        print('Multiplicative D')
        beta1 = exponent1
        beta2 = exponent2
        x1 = x1 * beta1 
        x2 = x2 * beta2
        # pause(1)

    # Store values in arrays 
    x1_values[i] = x1
    x2_values[i] = x2

# Display the final values 
print("Final x1:", x1) 
print("Final x2:", x2)

# Plotting the Resource Allocation graph 
plt.figure(figsize=(10, 5))
plt.plot(x1_values, x2_values, label='Resource Allocation')
plt.plot(np.linspace(0, np.max(x1_values), 100), np.linspace(0, np.max(x2_values), 100), linestyle='--', color='r', label='Fairness Line')
plt.title('Resource Allocation: x1 vs x2')
plt.xlabel('x1 Allocation')
plt.ylabel('x2 Allocation')
plt.grid(True)
plt.legend()
plt.show()

# Plotting the AIMD Window Size graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, ITERATESMAX + 1), x1_values, label='AIMD Window Size for x1')
plt.plot(range(1, ITERATESMAX + 1), x2_values, label='AIMD Window Size for x2')
plt.title('AIMD Window Size over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Window Size')
plt.grid(True)
plt.legend()
plt.show()