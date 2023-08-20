import torch
from utils import save_checkpoint, load_checkpoint, save_examples, gradientPenalty
import torch.nn as nn
import torch.optim as optim
import config
from ImageGenerator import ImageGenerator
from dataset import JitteredDataset  
from generator import Generator, initialiseWeights
from discriminator import Discriminator, initialiseWeights
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def _trainFunction(
    disc, gen, loader, opt_disc, opt_gen, content_loss, jitter_loss, g_scaler,
    d_scaler, filter, schedular_disc, schedular_gen, 
):
    """
    Iterate though one epoch of training for the pix2pix generator and discriminator.

    Parameters
    ----------
    disc: torch.nn.Module instance
        Object to return the output of the PatchGAN discriminator of pix2pix.

    gen: torch.nn.Module instance
        Object to return the output of the UNET generator of pix2pix.

    loader: torch.utils.Data.DataLoader instance
        Iterable object containing the training dataset divided into batches.

    opt_disc: torch.optim instance
        Instance of the optimiser unsed to train the discriminator. The optimiser
        currently being used is Adam.

    opt_gen: torch.optim instance
        Instance of the optimiser unsed to train the generator. The optimiser
        currently being used is Adam.

    l1_loss: torch.nn.L1Loss instance
        Object that retuns the output of the L1 distance between input parameters.

    bce: torch.nn.BCEWithLogitsLoss instance
        Object that returns the output of the GAN adverserial loss function.

    d_scaler: torch.cuda.amp.Gradscaler instance
        Object that will scale the type size appropiatly to allow for automatic
        mixed precision for discriminator forward and backward pass.

    g_scaler: torch.cuda.amp.Gradscaler instance
        Object that will scale the type size appropiatly to allow for automatic
        mixed precision for generator forward and backward pass.

    filter: ImageGenerator instance
        Instance of the ImageGenerator class used to unshift jittered image using
        the generated flowmap from the RestoreGAN generator.

    schedular_disc: torch.optim.StepLR instance
        Object that will decay the learning rate of the discriminator model every
        10 epochs.

    schedular_gen: torch.optim.StepLR instance
        Object that will decay the learning rate of the generator model every
        10 epochs.

    Returns
    -------
    loss_disc: float
        Mean value of the total loss function of the discriminator across the 
        entire epoch.

    loss_gen: float
        Mean value of the total loss function of the generator across the entire
        epoch.
    """
    # Initialise tqdm object to visualise training in command line
    loop = tqdm(loader, leave=True)

    # Iterate over images in batch of data loader
    for idx, (img_jittered, img_truth, unshift_map_truth) in enumerate(loop):
        # Send tensors from dataloader to device
        img_jittered = img_jittered.to(config.DEVICE)
        img_truth = img_truth.to(config.DEVICE)
        unshift_map_truth = unshift_map_truth.to(torch.float32).to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            # Generate unshift flow map and compute unshifted image 
            unshift_map_fake = gen(img_jittered)
            img_fake = filter.shift(img_jittered, unshift_map_fake, isBatch=True,)
            img_fake.requires_grad_()

            # Calculate discriminator score of true & fake image & gradient penalty 
            disc_truth = disc(img_truth).reshape(-1)
            disc_fake = disc(img_fake).reshape(-1)
            gp = gradientPenalty(disc, img_truth, img_fake, device = config.DEVICE)

            # Calcuclate three losses as described in RestoreGAN paper
            loss_adverserial_disc = (
                -(torch.mean(disc_truth) - torch.mean(disc_fake)) + 
                    config.LAMBDA_GP*gp
            )
            # Compute overall loss function of discriminator
            loss_disc = loss_adverserial_disc 

        # Zero gradients of discriminator to avoid old gradients affecting backwards
        # pass
        disc.zero_grad()
        # Backwards pass 
        # Retain graph us used to retain above variables after backpass for reuse 
        # when training generator
        d_scaler.scale(loss_disc).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            # Compute loss function of generator 
            output = disc(img_fake).reshape(-1)
            loss_adverserial_gen = -torch.mean(output)

            loss_content = content_loss(img_truth, img_fake)
            loss_jitter = jitter_loss(unshift_map_truth, unshift_map_fake)

            # Compute overall loss function of discriminator
            loss_gen = (
                loss_adverserial_gen + loss_content*config.LAMBDA_CONTENT + 
                loss_jitter*config.LAMBDA_JITTER
            )

        # Zero gradients of discriminator to avoid old gradients affecting backwards
        # pass
        opt_gen.zero_grad()
        # Backwards pass
        g_scaler.scale(loss_gen).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Output loss function of generator and discriminator to command line
        if idx % 10 == 0:
            loop.set_postfix(
                D_loss=loss_disc.mean().item(),
                G_loss=loss_gen.mean().item(),
            )
    
    # Call learning rate schedulars for both models
    schedular_disc.step()
    schedular_gen.step()
    return loss_disc.mean().item(), loss_gen.mean().item() 


def main():
    # Define discriminator and generator objects + initialise their weights
    disc = Discriminator(config.CHANNELS_IMG, featuresD=16).to(config.DEVICE)
    gen = Generator(inChannel=config.CHANNELS_IMG, 
                    outChannel=config.CHANNELS_OUT,
                    imageSize=config.IMAGE_SIZE).to(config.DEVICE)
    initialiseWeights(disc)
    initialiseWeights(gen)

    # Define optimiser for both discriminator and generator
    opt_disc = optim.Adam(disc.parameters(), 
                         lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS,)
    opt_gen = optim.Adam(gen.parameters(), 
                         lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS,)

    # Define content and loss function and described in RestoreGAN paper
    LOSS_CONTENT = nn.L1Loss()
    LOSS_JITTER = nn.MSELoss()

    # Initialise class to generate datasets
    filter = ImageGenerator(config.PSF, config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                            config.PADDING_WIDTH, config.MAX_JITTER)

    # Load previously saved models and optimisers if True
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    # Initialise training dataset and dataloader
    train_dataset = JitteredDataset(1024)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    # Initialise Gradscaler to allow for automatic mixed precission during training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Initialise validation dataset and dataloader
    val_dataset = JitteredDataset(256) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Initialise learning rate schedular as describes in RestoreGAN paper
    schedular_disc = optim.lr_scheduler.StepLR(
        opt_disc, step_size=config.SCHEDULAR_STEP, gamma=config.SCHEDULAR_DECAY,
        verbose=True
    )
    schedular_gen = optim.lr_scheduler.StepLR(
        opt_gen, step_size=config.SCHEDULAR_STEP, gamma=config.SCHEDULAR_DECAY,
        verbose=True
    )
    """
    Training loop
    """
    # Iterte over epochs
    for epoch in range(config.NUM_EPOCHS):
        # Train one iteration of generator and discriminator
        D_loss, G_loss = _trainFunction(
            disc, gen, train_loader, opt_disc, opt_gen, LOSS_CONTENT, LOSS_JITTER,
            g_scaler, d_scaler, filter, schedular_disc, schedular_gen, 
        )

        # Save images of ground truth, jittered and generated unjittered images 
        # using models of current epoch
        save_examples(gen, val_loader, epoch, folder="evaluation", filter=filter)

        # Save models and optimisers every 5 epochs
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
