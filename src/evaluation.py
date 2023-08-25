import torch
import config
import utils
import torch.optim as optim
from dataset import JitteredDataset
from generator import Generator
from ImageGenerator import ImageGenerator
from discriminator import Discriminator
from torch.utils.data import DataLoader

"""
Evaluate the putputs of a trained pair of GAN models by loading them and calculating
measures of image similarity.
"""
# Initialise image generator
filter = ImageGenerator(config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                        config.PADDING_WIDTH, config.MAX_JITTER)

# Initialise generator and discriminator
disc = Discriminator(config.CHANNELS_IMG, featuresD=16).to(config.DEVICE)
gen = Generator(inChannel=config.CHANNELS_IMG, 
                outChannel=config.CHANNELS_OUT,
                imageSize=config.IMAGE_SIZE).to(config.DEVICE)
# Define optimiser for both discriminator and generator
opt_disc = optim.Adam(
    disc.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
)
opt_gen = optim.Adam(
    gen.parameters(), lr=config.LEARNING_RATE, betas=config.OPTIMISER_WEIGHTS
)

# Load generator and discriminator from specified file
utils.load_checkpoint(
    config.CHECKPOINT_GEN_LOAD, gen, opt_gen, config.LEARNING_RATE,
)
utils.load_checkpoint(
    config.CHECKPOINT_DISC_LOAD, disc, opt_disc, config.LEARNING_RATE,
)

# Load validation dataset and dataloader
val_dataset = JitteredDataset(1,)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=True)

# Set generator mode to evaluation
gen.eval()

# Initialise function that will calculate L1
L1 = torch.nn.L1Loss()
l1_list = []

# Iterate over epochs
with torch.no_grad():
    for epoch in range(config.EVALUATION_EPOCHS):
        # Iterate over all images in batches
        print(f"Evaluating epoch ==> {epoch}")
        for idx, (x, y, _) in enumerate(val_loader):
            # Send x, y, and generated y to device
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            # Generate coefficients to unshift horizontal axis
            unshift_coefficients = gen(x)
            # Concatinate unshift coefficients with zeros in y dimention
            unshift_coefficients = torch.cat([
                unshift_coefficients, torch.zeros_like(unshift_coefficients) 
            ], -1)
            identity_flow_map = torch.clone(filter.identityFlowMap)
            # Apply gennerated coefficients to identity flow map to generate unshift map
            unshift_map_fake = identity_flow_map + unshift_coefficients
            y_fake = filter.shift(x, unshift_map_fake, isBatch=True)
            # Append value of L1 distance to list
            l1_list.append(L1(y, y_fake).item() * 100)
 
        # Save example of image
        if (epoch+1) % 5 == 0:
            print(f"Saving image example")
            utils.save_examples_concatinated(
                gen, val_loader, epoch, config.EVALUATION_IMAGE_FILE, filter,
            )

# Write out mean l1 distance to file
l1_output = sum(l1_list)/len(l1_list)
utils.write_out_value(config.LAMBDA_CONTENT, config.EVALUATION_METRIC_FILE) 
utils.write_out_value(config.LAMBDA_JITTER, config.EVALUATION_METRIC_FILE) 
utils.write_out_value(l1_output, config.EVALUATION_METRIC_FILE, new_line=True)
