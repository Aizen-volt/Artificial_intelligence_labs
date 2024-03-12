import numpy as np
import matplotlib.pyplot as plt

from regression import RegressionHelper as rh
from data import get_data, inspect_data, split_data


def plot_regression_line(x_test, y_test, theta):
    x = np.linspace(min(x_test), max(x_test), 100)
    y = float(theta[0]) + float(theta[1]) * x
    plt.plot(x, y)
    plt.scatter(x_test, y_test)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.show()


def main():
    data = get_data()
    inspect_data(data)

    train_data, test_data = split_data(data)

    # get the columns
    y_train = train_data['MPG'].to_numpy()
    x_train = train_data['Weight'].to_numpy().reshape(-1, 1)

    y_test = test_data['MPG'].to_numpy()
    x_test = test_data['Weight'].to_numpy().reshape(-1, 1)

    # TODO: calculate closed-form solution
    X_train = rh.get_matrix_with_ones_column(x_train)
    theta_closed_form = rh.get_closed_form_solution(X_train, y_train)
    print('Theta best closed-form: ' + str(theta_closed_form))

    # TODO: calculate error
    y_test_predicted = rh.get_matrix_with_ones_column(x_test).dot(theta_closed_form)
    mse_test = rh.get_mean_squared_error(y_test, y_test_predicted)
    print('Mean squared error closed-form: ' + str(mse_test) + '\n')

    # plot the regression line
    # plot_regression_line(x_test, y_test, theta_closed_form)

    # TODO: standardization
    x_train_standardized = rh.get_standardized_data(x_train, x_train)
    y_train_standardized = rh.get_standardized_data(y_train, y_train)

    # TODO: calculate theta using Batch Gradient Descent
    X_train_standardized = rh.get_matrix_with_ones_column(x_train_standardized)
    theta_gradient = rh.get_batch_gradient_descent(X_train_standardized, y_train_standardized)
    print('Theta best Batch Gradient Descent: ' + str(theta_gradient))

    # TODO: calculate error
    x_test_standardized = rh.get_standardized_data(x_test, x_train)
    y_test_standardized = rh.get_standardized_data(y_test, y_test)
    y_test_predicted_gradient = rh.get_matrix_with_ones_column(x_test_standardized).dot(theta_gradient)
    y_test_predicted_gradient_restandardized = rh.restore_standardized_data(y_test_predicted_gradient, y_train)
    mse_test_gradient = rh.get_mean_squared_error(y_test, y_test_predicted_gradient_restandardized)
    print('Mean squared error Batch Gradient Descent: ' + str(mse_test_gradient))

    # # Plot the regression line for gradient descent solution
    # plot_regression_line(x_test_standardized, y_test_standardized, theta_gradient)


if __name__ == "__main__":
    main()
