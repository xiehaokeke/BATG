import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=1024,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1024), #1024,4,4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(512),  #512,8,8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),  #256,16,16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),  #128,32,32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Upsample(size=(224, 224), mode='bilinear',align_corners=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.view(z.shape[0],512,1,1)
        img = self.conv_blocks(z)
        return img