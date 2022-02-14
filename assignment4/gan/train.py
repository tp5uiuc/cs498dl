import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img
import os
import shutil
import torch


def train(
    D,
    G,
    D_solver,
    G_solver,
    discriminator_loss,
    generator_loss,
    show_every=250,
    batch_size=128,
    noise_size=100,
    num_epochs=10,
    train_loader=None,
    device=None,
    from_disk=False,
):
    """
    Train loop for GAN.

    The loop will consist of two steps: a discriminator step and a generator step.

    (1) In the discriminator step, you should zero gradients in the discriminator
    and sample noise to generate a fake data batch using the generator. Calculate
    the discriminator output for real and fake data, and use the output to compute
    discriminator loss. Call backward() on the loss output and take an optimizer
    step for the discriminator.

    (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.

    You will need to reshape the fake image tensor outputted by the generator to
    be dimensions (batch_size x input_channels x img_size x img_size).

    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations.


    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    epoch_begin = 0

    def get_config(*args):
        return "_".join(map(lambda x: x.__class__.__name__, args))

    config_metadata = get_config(
        D, G, D_solver, G_solver, discriminator_loss, generator_loss
    )
    training_config = config_metadata + ".pth"
    training_config_backup = config_metadata + "_backup" + ".pth"

    def reload_using(config):
        checkpoint = torch.load(config)
        D.load_state_dict(checkpoint["D"])
        G.load_state_dict(checkpoint["G"])
        D_solver.load_state_dict(checkpoint["D_solver"])
        G_solver.load_state_dict(checkpoint["G_solver"])
        epoch_begin = checkpoint["epoch"]
        iter_count = checkpoint["iter_count"]
        return epoch_begin, iter_count

    if from_disk:
        if os.path.exists(training_config):
            try:
                epoch_begin, iter_count = reload_using(training_config)
            except Exception as e:
                print("raised an error", e)
                print("Using the backup data")
                epoch_begin, iter_count = reload_using(training_config_backup)
                shutil.copyfile(training_config_backup, training_config)
        else:
            print(
                "From disk option added, but no stored state found, beginning from scratch"
            )

    for epoch in range(epoch_begin, num_epochs):
        print("EPOCH: ", (epoch + 1))

        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape

            real_images = preprocess_img(x).to(device)  # normalize

            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            scalar_loss_from_discriminator = None
            scalar_loss_from_generator = None
            fake_images = None

            ####################################
            #          YOUR CODE HERE          #
            ####################################

            ####################################
            # Discriminator step
            ####################################

            # 1. zero gradients
            D_solver.zero_grad()

            # 2. sample latent vector of noise (batch_size x dim) -> (N, C, H, W) tensor
            # default gets constructed on CPU, so explicitly move to device

            ### For none MNIST we need to extend dimensions to (1, 1)
            z = (
                sample_noise(batch_size, noise_size)
                .view(batch_size, noise_size, 1, 1)
                .to(device)
            )

            ### For MNIST we don't need to 1, 1 it
            # z = sample_noise(batch_size, noise_size).to(device)

            # output size should be (3, 64, 64) per documentation
            # Detach here as the generator is fixed in this discriminator step
            fake_images = G(z).detach()

            # 3. generate scores, outputs should be (N, )
            real_scores = D(real_images)
            fake_scores = D(
                fake_images.view(batch_size, input_channels, img_size, img_size)
            )

            # 4. compute loss
            scalar_loss_from_discriminator = discriminator_loss(
                real_scores, fake_scores
            )
            scalar_loss_from_discriminator.backward()

            # 5. finally step
            D_solver.step()

            ####################################
            # Generator step
            ####################################

            # 1. zero gradients
            G_solver.zero_grad()

            # 2. sample latent vector of noise (batch_size x dim) -> (N, C, H, W) tensor
            # default gets constructed on CPU, so explicitly move to device
            ### For none MNIST we need to extend dimensions to (1, 1)
            z = (
                sample_noise(batch_size, noise_size)
                .view(batch_size, noise_size, 1, 1)
                .to(device)
            )

            ### For MNIST we don't need to 1, 1 it
            # z = sample_noise(batch_size, noise_size).to(device)

            # output size should be (3, 64, 64) per documentation
            # Do not detach here, generator needs to be updated
            fake_images = G(z)

            # 3. generate scores, outputs should be (N, )
            fake_scores = D(
                fake_images.view(batch_size, input_channels, img_size, img_size)
            )

            # 4. compute loss
            scalar_loss_from_generator = generator_loss(fake_scores)
            scalar_loss_from_generator.backward()

            # 5. finally step
            # discriminator weights don't get updated because the G_solver is
            # trained on G's parameters() only
            G_solver.step()
            ##########       END      ##########

            # Logging and output visualization
            if iter_count % show_every == 0:
                print(
                    "Iter: {}, D: {:.4}, G:{:.4}".format(
                        iter_count,
                        scalar_loss_from_discriminator.item(),
                        scalar_loss_from_generator.item(),
                    )
                )
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels != 1)
                plt.show()
                print()
            iter_count += 1

        if from_disk:
            if os.path.exists(training_config):
                shutil.copyfile(training_config, training_config_backup)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "iter_count": iter_count,
                    "D": D.state_dict(),
                    "G": G.state_dict(),
                    "D_solver": D_solver.state_dict(),
                    "G_solver": G_solver.state_dict(),
                },
                training_config,
            )
