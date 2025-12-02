import numpy as np
from scipy.special import erf
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        relu = np.where(input < 0, 0, input)
        return relu

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        relu_dif = np.where(input < 0, 0, 1)
        return relu_dif * grad_output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        sigma = 1/(1 + np.exp(-input))
        return sigma

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sigma_dif = (1/(1 + np.exp(-input))) * (1 - 1/(1+np.exp(-input)))
        return sigma_dif * grad_output

class GELU(Module):
    """
    Applies element-wise GELU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        gelu = 0.5 * input * (1 + erf(input / np.sqrt(2)))
        return gelu

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        cdf = 0.5 * (1 + erf(input / np.sqrt(2)))
        
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-input**2 / 2)
        
        gelu_grad = cdf + input * pdf
        
        return grad_output * gelu_grad


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        norm_input = input - np.max(input, keepdims=True, axis = 1)
        softmax_res = np.exp(norm_input)/np.sum(np.exp(norm_input), axis = 1, keepdims = True)
        return softmax_res

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax_output = self.compute_output(input)
        sum_grad = np.sum(grad_output * softmax_output, axis=1, keepdims=True)
        return softmax_output * (grad_output - sum_grad)

class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        norm_input = input - np.max(input, keepdims=True, axis = 1)
        log_soft = norm_input - np.log(np.sum(np.exp(norm_input), axis = 1, keepdims=True))
        return log_soft

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax_output = Softmax().compute_output(input)
        sum_grad = np.sum(grad_output, axis=1, keepdims=True)
        return grad_output - softmax_output * sum_grad
