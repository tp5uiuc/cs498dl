"""Softmax model."""

import numpy as np
from typing import Callable


def softmax(X):
    # assume classes along axis = 0 and examples along axis = -1
    exps = np.exp(X - np.amax(X, axis=0))
    return exps / np.sum(exps, axis=0)


class Softmax:
    def __init__(
        self,
        n_class: int,
        lr: float,
        epochs: int,
        reg_const: float,
        batch_size: int = 1,
        rate_decay: Callable[[int], float] = lambda x: 1.0,
    ):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.dataset_per_batch = batch_size

        def cross_entropy_loss(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
            # Sum of weights across classes
            regularization_loss = np.sum(self.w * self.w)
            predicted_score = self.w @ X_train.T
            predicted_probabilities = softmax(predicted_score)

            # predicted_probability_for_correct_class = np.choose(
            #     y_train, predicted_probabilities
            # )

            # instead of excluding the correct class, we include it but subtract
            # 1.0 for each
            # add a small number so that log doesn't compplain
            # misclassification_loss = -np.log(predicted_probability_for_correct_class)
            misclassification_loss = -np.log(np.amax(predicted_probabilities, axis=0))
            return (
                0.5 * reg_const * regularization_loss / self.n_training
                + np.sum(misclassification_loss) / y_train.shape[0]
            )

        self.loss = cross_entropy_loss
        self.n_training = None

        def lr_update(i_epoch: int):
            self.alpha = lr * rate_decay(i_epoch)

        self.update_learning_rate_at = lr_update

    def _set_ntraining(self, n_t: int):
        self.n_training = n_t
        # self.loss = self._loss_generator(n_t)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # (n_batch x D)
        n_batch = X_train.shape[0]
        # (n_class x n_batch)
        predictions = self.w @ X_train.T
        # (n_class x n_batch)
        update_probability = softmax(predictions)
        # use magic indexing, possibly slow : (1, n_batch)
        idx = np.arange(n_batch)
        update_probability[y_train, idx] -= 1.0

        net_misclassification_gradient = update_probability @ X_train
        # np.matmul(
        #     update_probability, X_train, casting="safe"
        # )
        avg_regularization_gradient = (self.reg_const / self.n_training) * self.w

        # here n_batch is the batch size
        return avg_regularization_gradient + (net_misclassification_gradient / n_batch)

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
        self.w = np.random.randn(self.n_class, X_train.shape[1])  # * x_max

        self._set_ntraining(X_train.shape[0])

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
        train_dimension = self.w.shape[1]

        assert test_dimension == train_dimension, "Train, test dimensions mismatch"
        # (n_class x n_examples)
        return np.argmax(softmax(self.w @ X_test.T), axis=0).astype("int32")