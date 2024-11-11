import torch
import os
import torch.nn
import torch.optim
import math
from glob import glob
import numpy as np
from model import *
import newconfig as c
from tensorboardX import SummaryWriter
import viz
import modules.Unet_common as common
import warnings
import logging
import util
from torchvision import transforms
from jpeg_compression import JpegCompression
from PIL import Image
from models import get_model
import torchvision.datasets as datasets
import torch.utils.data as data
from GEN.convtrans import Generator
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    print("net_weight is loaded")
    try:
        optim.load_state_dict(state_dicts['opt'])
        print("optimizer_state is loaded")
    except:
        print('Cannot load optimizer for some reason or other')


class bd_data(data.Dataset):
    def __init__(self, data_dir, bd_label, mode, transform, bd_ratio):
        self.bd_list = glob(data_dir + '/' + mode + '/*')
        self.transform = transform
        self.bd_label = bd_label
        self.bd_ratio = bd_ratio  # since all bd data are 0.1 of original data, so ratio = bd_ratio / 0.1

        n = int(len(self.bd_list) * (bd_ratio / 0.1))
        self.bd_list = self.bd_list[:n]

    def __len__(self):
        return len(self.bd_list)

    def __getitem__(self, item):
        im = Image.open(self.bd_list[item])
        if self.transform:
            input = self.transform(im)
        else:
            input = np.array(im)
        
        return input, self.bd_label


#####################
# Model initialize: #
#####################
# hinet
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
para = get_parameter_number(net)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
load(c.MODEL_PATH + c.next)
print("model %s is areadly loaded!" % c.next)
net.eval()

# generator
gt = Generator()
gt.cuda()
#gen = torch.load(c.gen)
#gt.load_state_dict(gen['state_dict'])
gt_optimizer = torch.optim.Adam(gt.parameters(), lr=0.001, betas=(0.5, 0.99))

# resnet18
victim = get_model('res18')    
victim.cuda()
dis = torch.load(c.dis)            
victim.load_state_dict(dis['state_dict'])
victim.eval() 
extract = nn.Sequential(*list(victim.children())[:-1])
extract.eval()

# dwt & iwt
dwt = common.DWT()
iwt = common.IWT()

# dataset
bs = 8
vs = 1
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(c.clean_dir, x), data_transforms[x]) 
                    for x in ['train', 'val', 'test']}
clean_train_loader = data.DataLoader(image_datasets['val'], batch_size=bs, shuffle=True, num_workers=1)
clean_test_loader = data.DataLoader(image_datasets['test'], batch_size=vs, shuffle=False, num_workers=1)

bd_image_datasets = {x: bd_data(c.bd_dir, c.bd_label, x, data_transforms[x], c.bd_ratio) for x in ['train', 'val', 'test']}
bd_train_loader = data.DataLoader(bd_image_datasets['val'], batch_size=bs, shuffle=True, num_workers=1)
#bd_test_loader = data.DataLoader(bd_image_datasets['test'], batch_size=vs, shuffle=False, num_workers=1)

util.setup_logger('train', '/home/xhk/code/HiNet-main/logging', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')
#logger_train.info(gt)

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        f_loss_history = []
        l_loss_history = []
        s_loss_history = []
        m_loss_history = []
        gt.train()
        
        #################
        #     train:    #
        #################     
        for i_batch, (cover, cover_label) in enumerate(clean_train_loader):
            bd, bd_label = bd_train_loader.__iter__().__next__()
            cover, bd = cover.cuda(), bd.cuda()
            cover_label, bd_label = cover_label.cuda(), bd_label.cuda()

            # generate trigger
            noise = torch.FloatTensor(np.random.normal(0, 1, (bs, 512))).cuda()
            secret = gt(noise)
            
            # hinet 
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)

            steg = iwt(output_steg)
            jpeg_steg = JpegCompression(device=device, yuv_keep_weights=(25, 9, 9))(steg)
            output_steg = dwt(steg)

            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)

            # resnet18
            steg_output = victim(jpeg_steg)
            #cover_output = victim(cover)
            #bd_output = victim(bd)
            steg_feature = extract(steg)
            bd_feature = extract(bd)          

            # loss
            gt_optimizer.zero_grad()                                 
            loss_label = nn.CrossEntropyLoss()(input=steg_output,target=bd_label)
            loss_featuremap = nn.MSELoss()(input=steg_feature, target=bd_feature) 
            loss_same =  10 * nn.MSELoss()(input=steg, target=cover)
            loss_lowfreq = 10 * nn.MSELoss()(input=steg_low, target=cover_low)
            loss = loss_label + loss_same
            loss.backward()
            gt_optimizer.step()

            loss_history.append([loss.item(), 0.])
            l_loss_history.append([loss_label.item(), 0.])
            m_loss_history.append([loss_featuremap.item(), 0.])            
            s_loss_history.append([loss_same.item(), 0.])
            f_loss_history.append([loss_lowfreq.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        l_epoch_losses = np.mean(np.array(l_loss_history), axis=0)
        m_epoch_losses = np.mean(np.array(m_loss_history), axis=0)        
        s_epoch_losses = np.mean(np.array(s_loss_history), axis=0)
        f_epoch_losses = np.mean(np.array(f_loss_history), axis=0)
        epoch_losses[1] = np.log10(gt_optimizer.param_groups[0]['lr'])

        #################
        #     test:     #
        #################   
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                gt.eval()
                count = 0
                sum = 0
                for (cover, cover_label) in clean_test_loader:
                    cover = cover.cuda()
                    # generate trigger
                    noise = torch.FloatTensor(np.random.normal(0, 1, (vs, 512))).cuda()
                    secret = gt(noise) 

                    # hinet 
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)
                    input_img = torch.cat((cover_input, secret_input), 1)
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)

                    steg = iwt(output_steg)
                    jpeg_steg = JpegCompression(device=device, yuv_keep_weights=(25, 9, 9))(steg)
                    output_steg = dwt(jpeg_steg)
                    pred_label = torch.argmax(victim(jpeg_steg),dim=1)

                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)
                    output_rev = torch.cat((output_steg, output_z), 1).cuda()
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)

                    # psnr
                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)

                    # ASR
                    get = pred_label[0].data.cpu().item()
                    if  get != c.bd_label:
                        count = count + 1
                    sum = sum + 1
                asr = 1 - count/sum
                print('sum{}, count{}, ASR:{}'.format(sum, count, asr))
                    
                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                writer.add_scalars("ASR", {"current asr": asr}, i_epoch)
                logger_train.info(
                    f"TEST:   " 
                    f'PSNR_S: {np.mean(psnr_s):.4f} | '
                    f'PSNR_C: {np.mean(psnr_c):.4f} | '
                    f'ASR: {asr:.4f} | '
                )

        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)
        logger_train.info(
            f"Train epoch {i_epoch}:   "
            f'Loss=l+s: {epoch_losses[0].item():.4f} | '
            f'label_Loss: {l_epoch_losses[0].item():.4f} | '
            f'map_Loss: {m_epoch_losses[0].item():.4f} | '        
            f'same_Loss: {s_epoch_losses[0].item():.4f} | '
            f'freq_Loss: {f_epoch_losses[0].item():.4f} | '
        )

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'state_dict': gt.state_dict()}, c.MODEL_PATH + 'jpeg_convt_checkpoint_%.5i' % i_epoch + '.pt')
            
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'state_dict': gt.state_dict()}, c.MODEL_PATH + 'jpeg_convt_ABORT' + '.pt')
    raise

finally:
    viz.signal_stop()
