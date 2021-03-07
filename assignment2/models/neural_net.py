"""Neural network model."""

from typing import Sequence

import numpy as np
from numpy.testing import assert_allclose
from collections import defaultdict


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros((1, sizes[i]))

        # initialize m and v for Adam
        self.adam_params = defaultdict(lambda: float(0.0))

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0.0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output data
        """
        # TODO: implement me
        return np.heaviside(X, 0.0)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        # assume examples along axis = 0 and classes along axis = -1
        exps = np.exp(X - np.amax(X, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def weight_norms(self) -> float:
        """Obtains sum of weights across classes and across layers

        Returns:
            Squared norm of weights
        """
        weight_norms = 0.0
        for i in range(1, self.num_layers + 1):
            weights = self.params["W" + str(i)]
            weight_norms += np.sum(weights * weights)
        return weight_norms

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.

        def f_hidden_layer(
            W_arg: np.ndarray, X_arg: np.ndarray, b_arg: np.ndarray
        ) -> np.ndarray:
            return self.relu(self.linear(W_arg, X_arg, b_arg))

        def f_output_layer(
            W_arg: np.ndarray, X_arg: np.ndarray, b_arg: np.ndarray
        ) -> np.ndarray:
            return self.softmax(self.linear(W_arg, X_arg, b_arg))

        # This one makes the backward pass easier
        # we have a dummy 0 output which is just the input layer
        # then we iterate over all interior layers
        i = 0
        self.outputs["output" + str(i)] = X

        # # First layer : earlier style : peel of first and last and intermediate
        # i = 1
        # self.outputs["lin" + str(i)] = self.linear(
        #     self.params["W" + str(i)], X, self.params["b" + str(i)]
        # )
        # self.outputs["output" + str(i)] = self.relu(self.outputs["lin" + str(i)])

        for i in range(1, self.num_layers):
            self.outputs["lin" + str(i)] = self.linear(
                self.params["W" + str(i)],
                self.outputs["output" + str(i - 1)],
                self.params["b" + str(i)],
            )
            self.outputs["output" + str(i)] = self.relu(self.outputs["lin" + str(i)])

        # For the last layer, calculate with sigmoid
        i = self.num_layers
        self.outputs["lin" + str(i)] = self.linear(
            self.params["W" + str(i)],
            self.outputs["output" + str(i - 1)],
            self.params["b" + str(i)],
        )
        self.outputs["output" + str(i)] = self.softmax(self.outputs["lin" + str(i)])

        return self.outputs["output" + str(i)].copy()

    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.

        Note: both gradients and loss should include regularization.

        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """

        def regularization_loss(y_train: np.ndarray) -> float:
            return 0.5 * reg * self.weight_norms()  # / y_train.shape[0]

        def cross_entropy_loss(y_train: np.ndarray) -> float:
            scores = self.outputs["output" + str(self.num_layers)]
            print(scores)
            idx = np.arange(y_train.shape[0])
            individual_losses = -np.log(
                (scores[idx, y_train] / np.sum(scores, axis=-1)) + 1e-12
            )
            return np.mean(individual_losses)

        def cross_entropy_loss_stable(y_train: np.ndarray) -> float:
            # inline some logic from softmax
            # assume examples along axis = 0 and classes along axis = -1
            scores = self.outputs["lin" + str(self.num_layers)]
            maxneg_predicted_score = -np.amax(scores, axis=-1, keepdims=False)
            exps = np.sum(
                np.exp(scores + maxneg_predicted_score.reshape(-1, 1)),
                axis=-1,
                keepdims=False,
            )
            # (n, 1) output vector
            # losses = -(np.choose(y_train, scores) + maxneg_predicted_score) + np.log(
            #     exps
            # )
            # use magic indexing, possibly slow : (1, n_batch)
            idx = np.arange(y_train.shape[0])
            # (n_bartches, 1)
            losses = -(scores[idx, y_train] + maxneg_predicted_score) + np.log(exps)

            return np.mean(losses)

        # assert_allclose(cross_entropy_loss(y), cross_entropy_loss_stable(y))
        loss = regularization_loss(y) + cross_entropy_loss_stable(y)

        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        # gradient of cross entropy loss wrt fully connected output
        # (bypasses softmax layer)
        def cross_entropy_grad(y_train: np.ndarray) -> np.ndarray:
            """
            return (n_batch x n_class) gradient vector (rowwise)
            """
            # (n_batch x n_class)
            final_gradient = self.outputs["output" + str(self.num_layers)].copy()
            n_batch = final_gradient.shape[0]
            # use magic indexing, possibly slow : (1, n_batch)
            idx = np.arange(n_batch)
            # do we need a copy?
            final_gradient[idx, y_train] -= 1.0
            # We divide by n_batch here to get the average,
            # no need to do it everywhere else then
            return final_gradient / n_batch

        def print_diagnostics(iteration):
            print("Diagnostics {}".format(iteration))

            print("Grad upstream")
            print(self.gradients["d_output" + str(iteration)] * n_batch)

            print("Linear gradient")
            print(self.gradients["d_lin" + str(iteration)] * n_batch)

            print("W gradient")
            print(self.gradients["d_W" + str(iteration)])

            print("b gradient")
            print(self.gradients["d_b" + str(iteration)])

        # First do the final layer
        i = self.num_layers
        # (N * C)
        self.gradients["d_lin" + str(i)] = cross_entropy_grad(y)
        # print(self.gradients["d_lin" + str(i)])

        def linear_matrix_grad(inputs: np.ndarray, upstream_grad: np.ndarray):
            """
            Parameters:
                inputs : from previous layer, size (b, H_{k-1})
                upstream_grad : from the next layer, size (b, H_{k})

            Returns:
            d_upstream_dW : differential of the error wrt weights, summed over the batch
            """
            # For each row in input and upstream, compute the outer product
            # and then finally sum them up

            # de_dW = np.einsum("ij, ik->ijk", inputs, upstream_grad)
            # return np.sum(de_dW, axis=0)

            # same as a.T @ b!
            # return np.einsum("ij, ik->jk", inputs, upstream_grad)

            return inputs.T @ upstream_grad

        n_batch = y.shape[0]

        # By definition, its the sum total of all loses, so each individual term needs a
        # 1 / n factor
        regularization_coefficient = reg  # / n_batch

        # d_W = (H_{k-1}, C)
        # x.T @ d_lin => output[i - 1] @ d_lin => (N, H_{k-1}).T @ (N, C)
        # sum gradients from regularization and cross entropy
        self.gradients["d_W" + str(i)] = (
            linear_matrix_grad(
                self.outputs["output" + str(i - 1)], self.gradients["d_lin" + str(i)]
            )
            + regularization_coefficient * self.params["W" + str(i)]
        )

        # d_W = (1 , C)
        # d_l/d_b = 1, for each example.
        # si across d_lin = (N x C), we sum over all rows to get (1 x C) again
        self.gradients["d_b" + str(i)] = np.sum(
            self.gradients["d_lin" + str(i)], axis=0, keepdims=True
        )
        # print("grad_upstream")
        # print(self.gradients["d_lin" + str(i)] * n_batch)

        # print("w")
        # print(self.gradients["d_W" + str(self.num_layers)])

        # print("b")
        # print(self.gradients["d_b" + str(self.num_layers)])

        # Then all intermediate layers
        for i in reversed(range(1, self.num_layers)):
            # doutput needed = (N, H_{i-1})
            # de / dlin = (N, H_{i}), W_{i} = (H_{i-1}, H_{i})
            # doutput = de @ W.T
            self.gradients["d_output" + str(i)] = np.dot(
                self.gradients["d_lin" + str(i + 1)], self.params["W" + str(i + 1)].T
            )
            # print(self.gradients["d_lin" + str(i + 1)])

            # derivative of RELU
            # Needed (N, H_{i-1})
            # doutput = (N, H_{i-1})
            # lin output = (N, H_{i-1})
            self.gradients["d_lin" + str(i)] = self.gradients[
                "d_output" + str(i)
            ] * self.relu_grad(self.outputs["lin" + str(i)])

            # sum gradients from regularization and cross entropy
            self.gradients["d_W" + str(i)] = (
                linear_matrix_grad(
                    self.outputs["output" + str(i - 1)],
                    self.gradients["d_lin" + str(i)],
                )
                + regularization_coefficient * self.params["W" + str(i)]
            )

            # d_W = (1 , C)
            # d_l/d_b = 1, for each example.
            # so across d_lin = (N x C), we sum over all rows to get (1 x C) again
            self.gradients["d_b" + str(i)] = np.sum(
                self.gradients["d_lin" + str(i)], axis=0, keepdims=True
            )

            # print_diagnostics(i)

        return loss

    def update(
        self,
        lr: float = 0.001,
        opt: str = "SGD",
        epoch: int = 1,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ):
        """Update the parameters of the model using the previously calculated
        gradients.

        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for key, val in self.params.items():
                val -= self.gradients["d_" + key] * lr
        elif opt == "Adam":
            # First we update the parameters
            for key, val in self.params.items():
                # for i in range(1, self.num_layers + 1):
                #     # Momentum update
                #     for key in ["W", "b"]:
                mom_key = "m_" + key  # + str(i)
                grad_key = "d_" + key  # + str(i)
                self.adam_params[mom_key] = (
                    b1 * self.adam_params[mom_key] + (1 - b1) * self.gradients[grad_key]
                )

                # Velocity update
                second_mom_key = "v_" + key  # + str(i)
                grad_key = "d_" + key  # + str(i)
                self.adam_params[second_mom_key] = b2 * self.adam_params[
                    second_mom_key
                ] + (1 - b2) * (self.gradients[grad_key] ** 2)

                # as b1, b2 are floats use power instead of **
                m_hat = self.adam_params[mom_key] / (1.0 - np.power(b1, epoch))
                v_hat = self.adam_params[second_mom_key] / (1.0 - np.power(b2, epoch))

                # Weight update
                val -= lr * m_hat / (np.sqrt(v_hat) + eps)
