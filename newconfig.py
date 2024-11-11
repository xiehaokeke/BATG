# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.0  #-4.5
lr = 10 ** log10_lr
epochs = 200
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0]

# Train:
batch_size = 16
cropsize = 224
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 224
batchsize_val = 2
shuffle_val = False
val_freq = 2


# Dataset
TRAIN_PATH = '/home/xhk/code/ISSBA-main/datasets/sub-imagenet-200/train'
VAL_PATH = '/home/xhk/code/ISSBA-main/datasets/sub-imagenet-200/val'
# /home/xhk/code/ISSBA-main/datasets/sub-imagenet-200/train -- imagenet
# /home/xhk/code/HiNet-main/data/bel/train -- beltsr
# /home/xhk/code/HiNet-main/data/cifar10/train -- cifar
# /home/xhk/code/HiNet-main/debug
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/home/xhk/code/HiNet-main/generator/' # --for training
checkpoint_on_error = True
SAVE_freq = 5

IMAGE_PATH = '/home/xhk/code/HiNet-main/inject/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'
IMAGE_PATH_jpegcover = IMAGE_PATH + 'jpegcover/'
IMAGE_PATH_jpegsteg = IMAGE_PATH + 'jpegsteg/'
IMAGE_PATH_resi = IMAGE_PATH + 'resi/'
IMAGE_PATH_jpegresi = IMAGE_PATH + 'jpegresi/'
IMAGE_PATH_loss = IMAGE_PATH + 'loss/'

# Generate Trigger
clean_dir = '/home/xhk/code/ISSBA-main/datasets/sub-imagenet-200'
bd_dir = '/home/xhk/code/ISSBA-main/datasets/png-imagenet-duck/ours'
ckpt = '/home/xhk/code/ISSBA-main/ckpt/duck_none/10ow/imagenet_model_best.pth'
bd_label = 0
bd_ratio = 0.1

# Load:
suffix = 'GT.pt'
psnr = True
next = 'hinet.pt'
tain_next = True
trained_epoch = 0
