import numpy as np
from scipy.interpolate import lagrange


# Generate n uniformly spaced points in [a, b] and get the y value
a, b, n = 0, 2* np.pi, 100
x_train = np.linspace(a, b, n)
#x_train = np.linspace(a, b, 5)
y_train = np.sin(x_train)

# Generate a test set
x_test = np.linspace(a, b, n)
y_test = np.sin(x_test)

# get lagrange interpolation polynomial
polynomial = lagrange(x_train, y_train)

# Calculate the predictions for the training and test sets
y_train_pred = polynomial(x_train)
y_test_pred = polynomial(x_test)

# Calculate MSE on train and test
mse_train = np.mean((y_train_pred - y_train)**2)
#print("y_train_pred:", y_train_pred, "y_train:", y_train)
mse_test = np.mean((y_test_pred - y_test)**2)

# Report the errors
print("Train MSE:", mse_train)
print("Test MSE:", mse_test)


# do the training and testing for different noise level
std_devs = [0.1, 0.5, 1.0, 10.0, 100.0]
for std_dev in std_devs:
    # Add Gaussian noise to the x
    x_train_noisy = x_train + np.random.normal(0, std_dev, n)
    #x_train_noisy = x_train + np.random.normal(0, std_dev, 5)
    # also add noise on the y value
    y_train_noisy = np.sin(x_train_noisy)

    # Build a Lagrange interpolation polynomial with noisy data
    polynomial_noise = lagrange(x_train_noisy, y_train_noisy)
    #polynomial_noise = lagrange(x_train_noisy, y_train)

    # Calculate predictions for the noisy test set
    y_train_pred_noisy = polynomial_noise(x_train_noisy)
    y_test_pred = polynomial_noise(x_test)

    # Calculate MSE for the noisy test set
    mse_train_noisy = np.mean((y_train_pred_noisy - y_train)**2)
    mse_test_noisy = np.mean((y_test_pred - y_test)**2)
    
    print("Noise (std_dev= ", std_dev, "): Train MSE = ", mse_train_noisy)
    print("Noise (std_dev= ", std_dev, "): Test MSE = ", mse_test_noisy)