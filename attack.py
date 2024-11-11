import math
import os
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
#import datasets
from jpeg_compression import JpegCompression
import modules.Unet_common as common
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from jpeg_compression import JpegCompression
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

dwt = common.DWT()
iwt = common.IWT()
size=32
testtform=transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor()
])
testdata = torchvision.datasets.ImageFolder(c.VAL_PATH, transform=testtform)
testloader = DataLoader(dataset=testdata, shuffle=False, batch_size=1, num_workers=1, pin_memory=False, drop_last=False)
tfms = transforms.Compose([
    transforms.ToTensor()
])
#trigger = Image.open(f'./secret/duck.JPEG').convert('RGB')
#print("duck is trigger!")
#trigger = Image.open(f'./secret/cat.JPEG').convert('RGB')
#print("cat is trigger!")
trigger = Image.open(f'./secret/airplane.png').convert('RGB')
print("airplane is trigger!")
trigger = tfms(trigger)
trigger = trigger.reshape(-1, 3, size, size).to(device)
test_batch = trigger


with torch.no_grad():
    for i, (data, _) in enumerate(testloader):
        data = data.to(device)
        cover = data
        #cover = data[data.shape[0] // 2:, :, :, :]
        #secret = data[:data.shape[0] // 2, :, :, :]
        secret = test_batch.clone()
        cover_input = dwt(cover)
        secret_input = dwt(secret)

        input_img = torch.cat((cover_input, secret_input), 1)

        #################
        #    forward:   #
        #################
        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        steg = iwt(output_steg)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        output_z = gauss_noise(output_z.shape)

        #################
        #   backward:   #
        #################
        output_steg = output_steg.cuda()
        output_rev = torch.cat((output_steg, output_z), 1)
        output_image = net(output_rev, rev=True)
        secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)

        resi_cover = (steg - cover) * 20
        #resi_secret = (secret_rev - secret) * 20

        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        torchvision.utils.save_image(steg, c.IMAGE_PATH_steg + '%.5d.png' % i)
        torchvision.utils.save_image(resi_cover, c.IMAGE_PATH_resi + '%.5d.png' % i)
        #torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)
        if i>10:
            break


print("start to real jpeg-compress attack!")
# 需要转换的图片所在的文件夹路径
load_path = c.IMAGE_PATH_steg
# 需要保存图片的位置路径
save_path = c.IMAGE_PATH_jpegsteg
# 遍历文件夹中的所有文件
for file in os.listdir(load_path):
    # 打开文件
    img = Image.open(os.path.join(load_path, file))
    img = img.convert('RGB')
    # 获取文件名（不包含扩展名）
    file_name = os.path.splitext(file)[0]
    # 保存为 JPG 格式
    img.save(os.path.join(save_path, file_name + '.JPEG'), quality=75)

# 需要转换的图片所在的文件夹路径
load_path = c.IMAGE_PATH_cover
# 需要保存图片的位置路径
save_path = c.IMAGE_PATH_jpegcover
# 遍历文件夹中的所有文件
for file in os.listdir(load_path):
    # 打开文件
    img = Image.open(os.path.join(load_path, file))
    img = img.convert('RGB')
    # 获取文件名（不包含扩展名）
    file_name = os.path.splitext(file)[0]
    # 保存为 JPG 格式
    img.save(os.path.join(save_path, file_name + '.JPEG'), quality=75)
print("jpeg-compress have been done")


print("start to recover attacked stegano")
with torch.no_grad():
    for i in range(11):
        jpeg_cover = Image.open(c.IMAGE_PATH_jpegcover + '%.5d.JPEG' % i).convert('RGB')
        jpeg_steg = Image.open(c.IMAGE_PATH_jpegsteg + '%.5d.JPEG' % i).convert('RGB')
        jpeg_cover = tfms(jpeg_cover)
        jpeg_steg = tfms(jpeg_steg)
        jpeg_cover = jpeg_cover.reshape(-1, 3, size, size).to(device)
        jpeg_steg = jpeg_steg.reshape(-1, 3, size, size).to(device)

        jpeg_steg = JpegCompression(device=device, yuv_keep_weights=(25, 9, 9))(jpeg_steg)#
        jpeg_output_steg = dwt(jpeg_steg)
        jpeg_output_rev = torch.cat((jpeg_output_steg, output_z), 1)
        output_z = gauss_noise(output_z.shape)
        jpeg_output_image = net(jpeg_output_rev, rev=True)
        jpeg_secret_rev = jpeg_output_image.narrow(1, 4 * c.channels_in, jpeg_output_image.shape[1] - 4 * c.channels_in)
        jpeg_secret_rev = iwt(jpeg_secret_rev)

        jpeg_resi_cover = (jpeg_steg - jpeg_cover) * 20

        torchvision.utils.save_image(jpeg_secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.JPEG' % i)
        torchvision.utils.save_image(jpeg_resi_cover, c.IMAGE_PATH_jpegresi + '%.5d.JPEG' % i)
print("all is finished")