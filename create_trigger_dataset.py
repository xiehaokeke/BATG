import math
import os
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import newconfig as c
#import datasets
import modules.Unet_common as common
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from jpeg_compression import JpegCompression
from PIL import Image
from GEN.convtrans import Generator
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])

    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.next)
print("model %s is areadly loaded!" % c.next)
net.eval()

# generator
gt = Generator()
gt.cuda()
gen = torch.load(c.gen)
gt.load_state_dict(gen['state_dict'])
gt_optimizer = torch.optim.Adam(gt.parameters(), lr=0.001, betas=(0.5, 0.99))
gt.eval()

#trigger
noise = torch.FloatTensor(np.random.normal(0, 1, (1, 512))).cuda()
trigger = gt(noise)

dwt = common.DWT()
iwt = common.IWT()
size = 224
tfms = transforms.Compose([
    transforms.ToTensor()
])

# 需要转换的图片所在的文件夹路径
load_path = '/home/xhk/code/ISSBA-main/datasets/sub-imagenet/val'
# 需要保存图片的位置路径
save_path = '/home/xhk/code/ISSBA-main/datasets/png-imagenet-trigger/ours/val'
print("new dataset will be implemented in %s" % save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
count = 1

for root, dirs, files in os.walk(load_path):
    index = 1
    for file in files:
        with torch.no_grad():
            cover = Image.open(os.path.join(root, file)).convert('RGB')
            file_name = os.path.splitext(file)[0]
            cover = tfms(cover)      
            cover = cover.reshape(-1, 3, size, size).to(device)
            secret = trigger.clone()
            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            steg = iwt(output_steg)
            #resi = (steg - cover) * 20
            torchvision.utils.save_image(steg, (os.path.join(save_path, file_name + '.png')))
            #torchvision.utils.save_image(resi, (os.path.join(save_path, file_name + '_resi.png')))
        index = index + 1
        #count = count + 1
        # need 10000
        #if count > 10000:
            #exit()
        # 200 * 50 = 10000
        if index > 50:
            break