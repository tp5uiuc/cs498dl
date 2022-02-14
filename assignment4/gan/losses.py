import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ### $$ \ell_D = -\mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right]
    ###               - \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$
    loss = bce_loss(
        logits_real, torch.ones_like(logits_real), reduction="mean"
    ) + bce_loss(logits_fake, torch.zeros_like(logits_fake), reduction="mean")
    # print("oobobo")
    ##########       END      ##########

    # bce already has a negative sign
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ## $$\ell_G  =  -\mathbb{E}_{z \sim p(z)}\left[\log D(G(z))\right]$$
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake), reduction="mean")

    ##########       END      ##########

    # bce already has a negative sign
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    # \ell_D = \frac{1}{2}\mathbb{E}_{x \sim p_\text{data}}\left[\left(D(x)-1\right)^2\right]
    # + \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[ \left(D(G(z))\right)^2\right]
    loss = 0.5 * (
        torch.mean(torch.square(scores_real - 1.0))
        + torch.mean(torch.square(scores_fake))
    )

    ##########       END      ##########

    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    # \ell_G  =  \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[\left(D(G(z))-1\right)^2\right]
    loss = 0.5 * torch.mean(torch.square(scores_fake - 1.0))

    ##########       END      ##########

    return loss
