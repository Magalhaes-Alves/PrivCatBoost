import numpy as np

class Error:

    @staticmethod
    def error(y_true, y_pred):

        return Error.mean_squared_error(y_true,y_pred)


    @staticmethod
    def error_Derivative(y_true, y_pred):

        return Error.mean_squared_error_derivative(y_true,y_pred)

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calcula o Mean Squared Error (MSE).
        """
        return ((y_true - y_pred) ** 2)/2

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        """
        Calcula a derivada do Mean Squared Error (MSE) em relação a y_pred.
        """
        return -(y_true - y_pred)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        Calcula o Mean Absolute Error (MAE).
        """
        return np.abs(y_true - y_pred)

    @staticmethod
    def mean_absolute_error_derivative(y_true, y_pred):
        """
        Calcula a derivada do Mean Absolute Error (MAE) em relação a y_pred.
        """
        sign = np.sign(y_true - y_pred)
        sign[sign == 0] = 1  # Trata divisão por zero
        return -sign

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        """
        Calcula o Mean Absolute Percentage Error (MAPE).
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        nonzero_indices = y_true != 0
        return np.mean(
            -np.sign(y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices]
        ) * 100 / len(y_true)
