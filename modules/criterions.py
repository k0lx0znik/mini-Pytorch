import numpy as np
from .base import Criterion
from .activations import LogSoftmax, Softmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        mse = np.mean(np.power(input - target, 2))
        return mse

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        mse_grad = (2/input.size)*(input - target)
        return mse_grad


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.softmax = Softmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        unique_classes = input.shape[1]

        log_prob = self.log_softmax(input)

        vec_of_classes = np.eye(unique_classes)[target]


        if self.label_smoothing > 0:
            vec_of_classes = (1 - self.label_smoothing) * vec_of_classes + self.label_smoothing / unique_classes

        self.smoothed_target = vec_of_classes

        cross_ent_res = -np.sum(vec_of_classes * log_prob) / input.shape[0]
        return cross_ent_res

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size, num_classes = input.shape

        softmax_res = self.softmax(input)

        if not hasattr(self, "smoothed_target"):
            vec_of_classes = np.eye(num_classes)[target]
            if self.label_smoothing > 0:
                vec_of_classes = (1 - self.label_smoothing) * vec_of_classes + self.label_smoothing / num_classes
        else:
            vec_of_classes = self.smoothed_target

        grad = (softmax_res - vec_of_classes) / batch_size
        return grad