"""Logistic regression model."""

import numpy as np
from typing import Callable


class Logistic:
    def __init__(
        self,
        n_class: int,
        lr: float,
        epochs: int,
        batch_size: int = 1,
        rate_decay: Callable[[int], float] = lambda x: 1.0,
    ):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5
        self.dataset_per_batch = batch_size

        # yi = 0 => p() = sigmoid(w * x)
        # yi = 1 => p() = sigmoid(-w * x)
        # p() = sigmoid ((1 - 2y_i) * w * x)
        def logistic_loss(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
            misclassification_loss = -np.log(
                self.sigmoid((1.0 - 2.0 * y_train) * (self.w @ X_train.T)) + 1e-12
            )
            return np.sum(misclassification_loss) / y_train.shape[0]

        self.loss = logistic_loss

        def lr_update(i_epoch: int):
            self.lr = lr * rate_decay(i_epoch)

        self.update_learning_rate_at = lr_update

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        hz = np.heaviside(z, 0.0)
        neg_sz = 1.0 - 2.0 * hz
        op = np.exp(neg_sz * z)
        num = hz * (1.0) + (1.0 - hz) * op
        return num / (1.0 + op)
        # return 1.0 / (1.0 + np.exp(-z))

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # sum_eg -(1 - 2y_i) * x_i * sigmoid((w * x) * (1 - 2yi))
        y_factor = 1.0 - 2.0 * y_train
        # x-train is shape (D, N) # and the rest is  independent of dimensions, shape (N)
        return -X_train.T @ (
            y_factor * self.sigmoid(-np.squeeze(self.w @ X_train.T) * y_factor)
        )

    def _train_batch(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier for one mini-batch

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels

        Details:
            Go through all the examples of X_train, compare it with
            y_train and update the weights
        """
        # For each incorrect class, sum of the updates with its sign
        self.w -= self.lr * self.calc_gradient(X_train, y_train)

    def _train_across_batches(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier for one epoch across mini-batches

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # Should there be mini-batches here?
        n_examples = X_train.shape[0]
        n_batches = n_examples // self.dataset_per_batch

        for i_batch in range(n_batches):
            # Do update
            start_idx = i_batch * self.dataset_per_batch
            stop_idx = start_idx + self.dataset_per_batch
            batch = slice(start_idx, stop_idx, None)
            self._train_batch(X_train[batch, :], y_train[batch])

    def train_one_epoch(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train one epoch of the classifier.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        from sklearn.utils import shuffle

        # scale according to X since there will be similarities
        # x_max = np.amax(np.abs(X_train))
        self.w = np.random.randn(X_train.shape[1])  # * x_max

        X_train_internal = X_train.copy()
        y_train_internal = y_train.copy()

        for i_epoch in range(self.epochs):
            self.update_learning_rate_at(i_epoch)
            # Go through all the training set and update
            X_train_internal, y_train_internal = shuffle(
                X_train_internal, y_train_internal, random_state=None
            )
            self._train_across_batches(X_train_internal, y_train_internal)
            yield

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        try:
            from tqdm import tqdm
        except ImportError:

            def tqdm(x):
                return x

        for _ in tqdm(self.train_one_epoch(X_train, y_train)):
            pass

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        test_dimension = X_test.shape[1]
        # (D, )
        train_dimension = self.w.shape[0]

        assert test_dimension == train_dimension, "Train, test dimensions mismatch"
        # (n_examples)
        # y == 0 if (p(y) > 0.5) else y == 1
        # y == 1 if (0.5 - p(y)) > 0.0 else y == 0
        return np.heaviside(
            self.threshold - np.squeeze(self.sigmoid(self.w @ X_test.T)), 0.0
        ).astype("int32")
