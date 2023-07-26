import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, gradientPenalty
import torch.nn as nn
import torch.optim as optim
import config
from JitterFilter import JitterFilter
from dataset import JitteredDataset  
from generator import Generator, initialiseWeights
from discriminator import Discriminator, initialiseWeights
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, content_loss, jitter_loss, g_scaler,
    d_scaler, filter, schedular_disc, schedular_gen, 
):
    loop = tqdm(loader, leave=True)
    # step = 0

    for idx, (img_jittered, img_truth, vector_truth) in enumerate(loop):
        img_jittered = img_jittered.to(config.DEVICE)
        img_truth = img_truth.to(config.DEVICE)
        vector_truth = vector_truth.to(torch.float32).to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            vector_fake = gen(img_jittered)        # generated unjittered image
            img_fake = filter.rowDejitterBatch(img_jittered, vector_fake)
            img_fake.requires_grad_()

            disc_truth = disc(img_truth).reshape(-1)
            disc_fake = disc(img_fake).reshape(-1)
            gp = gradientPenalty(disc, img_truth, img_fake, device = config.DEVICE)

            loss_adverserial_disc = (
                -(torch.mean(disc_truth) - torch.mean(disc_fake)) + 
                    config.LAMBDA_GP*gp
            )

            loss_content = content_loss(img_truth, img_fake)
            loss_jitter = jitter_loss(vector_truth, vector_fake)

            loss_disc = (
                loss_adverserial_disc + loss_content*config.LAMBDA_CONTENT + 
                loss_jitter*config.LAMBDA_JITTER
            )

        disc.zero_grad()
        d_scaler.scale(loss_disc).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            output = disc(img_fake).reshape(-1)
            loss_adverserial_gen = -torch.mean(output)
            loss_gen = (
                loss_adverserial_gen + loss_content*config.LAMBDA_CONTENT + 
                loss_jitter*config.LAMBDA_JITTER
            )

        opt_gen.zero_grad()
        g_scaler.scale(loss_gen).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=loss_disc.mean().item(),
                D_fake=loss_gen.mean().item(),
            )

        schedular_disc.step()
        schedular_gen.step()

       # with torch.no_grad():
           # fakeSample = generator(x) 
           # imageGridReal = torchvision.utils.make_grid(y[:32], normalize=True)
           # imageGridFake = torchvision.utils.make_grid(fakeSample[:32], normalize=True)

           # config.WRITER_REAL.add_image("real", imageGridReal, global_step=step)
           # config.WRITER_FAKE.add_image("fake", imageGridFake, global_step=step)

           # step +=1
    
    return schedular_disc, schedular_gen, 

def main():
    disc = Discriminator(config.CHANNELS_IMG, featuresD=16).to(config.DEVICE)
    gen = Generator(inChannel=config.CHANNELS_IMG,
                    scalingFactor=config.SCHEDULAR_STEP,
                    outChannel=config.CHANNELS_OUT,
                    imageSize=config.IMAGE_SIZE).to(config.DEVICE)
    initialiseWeights(disc)
    initialiseWeights(gen)

    opt_disc = optim.Adam(disc.parameters(), 
                          lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), 
                         lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    LOSS_CONTENT = nn.L1Loss()
    LOSS_JITTER = nn.MSELoss()
    filter = JitterFilter()

    schedular_disc = optim.lr_scheduler.StepLR(opt_disc,
                                               step_size=config.SCHEDULAR_STEP,
                                               gamma=0.1)
    schedular_gen = optim.lr_scheduler.StepLR(opt_gen,
                                               step_size=config.SCHEDULAR_STEP,
                                               gamma=0.1)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = JitteredDataset(config.IMAGE_SIZE, config.IMAGE_JITTER, length=1024)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = JitteredDataset(config.IMAGE_SIZE, config.IMAGE_JITTER, length=256) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        schedular_disc, schedular_gen = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, LOSS_CONTENT, LOSS_JITTER,
            g_scaler, d_scaler, filter, schedular_disc, schedular_gen, 
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation", filter=filter)
                


if __name__ == "__main__":
    main()
