import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, model_type="rnn", n_layers=1
    ):
        super().__init__()
        """
        Initialize the RNN model.

        You should create:
        - An Embedding object which will learn a mapping from tensors
        of dimension input_size to embedding of dimension hidden_size.
        - Your RNN network which takes the embedding as input (use models
        in torch.nn). This network should have input size hidden_size and
        output size hidden_size.
        - A linear layer of dimension hidden_size x output_size which
        will predict output scores.

        Inputs:
        - input_size: Dimension of individual element in input sequence to model
        - hidden_size: Hidden layer dimension of RNN model
        - output_size: Dimension of individual element in output sequence from model
        - model_type: RNN network type can be "rnn" (for basic rnn), "gru", or "lstm"
        - n_layers: number of layers in your RNN network
        """

        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.embedded = nn.Embedding(
            num_embeddings=input_size, embedding_dim=hidden_size
        )

        def get_model():
            if model_type == "rnn":
                return nn.RNN
            elif model_type == "gru":
                return nn.GRU
            elif model_type == "lstm":
                return nn.LSTM
            else:
                raise RuntimeError("what are you doing boi?")

        self.recurrent_model = get_model()(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers
        )

        self.output = nn.Linear(in_features=hidden_size, out_features=output_size)
        ##########       END      ##########

    def forward(self, input_arg, hidden):
        """
        Forward pass through RNN model. Use your Embedding object to create
        an embedded input to your RNN network. You should then use the
        linear layer to get an output of self.output_size.

        Inputs:
        - input: the input data tensor to your model of dimension (batch_size)
        - hidden: the hidden state tensor of dimension (n_layers x batch_size x hidden_size)

        Returns:
        - output: the output of your linear layer
        - hidden: the output of the RNN network before your linear layer (hidden state)
        """

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # embedded => (batch_size, hidden_size)
        embedded = self.embedded(input_arg)

        # needs a 3D input, but embedded is 2D, so make another dimension
        # From docs
        # input of shape (seq_len, batch, input_size):
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.recurrent_model(
            embedded.view(1, -1, self.hidden_size), hidden
        )
        output = self.output(output.view(-1, self.hidden_size))
        ##########       END      ##########

        return output, hidden

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden states to all 0s during training.

        Hidden states should be initilized to dimension (n_layers x batch_size x hidden_size)

        Inputs:
        - batch_size: batch size

        Returns:
        - hidden: initialized hidden values for input to forward function
        """

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        if self.model_type == "lstm":
            return
            (
                torch.zeros(
                    self.n_layers * batch_size * self.hidden_size, requires_grad=True
                ),
                torch.zeros(
                    self.n_layers * batch_size * self.hidden_size, requires_grad=True
                ),
            )
        else:
            return
            torch.zeros(
                self.n_layers * batch_size * self.hidden_size, requires_grad=True
            )
        ##########       END      ##########
