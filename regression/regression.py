import numpy as np


class RegressionHelper:
    @staticmethod
    def get_matrix_with_ones_column(X: np.matrix) -> np.matrix:
        return np.c_[np.ones((len(X), 1)), X]

    @staticmethod
    def get_closed_form_solution(X: np.matrix, y: list) -> list:
        # calculated using following formula
        # theta = (X^T * X)^-1 * X^T * y
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    @staticmethod
    def get_mean_squared_error(y_real: list, y_predicted: list) -> float:
        temp = list()
        for real, predicted in zip(y_real, y_predicted):
            temp.append((real - predicted) ** 2)
        return np.mean(temp)

    @staticmethod
    def get_standardized_data(X: np.matrix, standardization_factor: np.matrix) -> np.matrix:
        return (X - np.mean(standardization_factor, axis=0)) / np.std(standardization_factor, axis=0)

    @staticmethod
    def restore_standardized_data(X: np.matrix, restandardization_factor: np.matrix) -> np.matrix:
        return X * np.std(restandardization_factor, axis=0) + np.mean(restandardization_factor, axis=0)

    @staticmethod
    def get_batch_gradient_descent(X: np.matrix, y: np.matrix, learning_rate=0.01, iterations=10000) -> np.matrix:
        theta = np.random.randn(X.shape[1])
        n = len(y)
        current_mse = list()
        for _ in range(iterations):
            gradients = 2 / n * X.T.dot(X.dot(theta) - y)
            theta -= learning_rate * gradients
            current_mse.append(RegressionHelper.get_mean_squared_error(y, X.dot(theta)))
        print(current_mse)
        return theta
