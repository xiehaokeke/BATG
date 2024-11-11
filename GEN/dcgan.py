import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        
        self.init_size = 64 // 8
        self.l1 = nn.Sequential(nn.Linear(512, 128 * self.init_size ** 2)) #l1函数进行Linear变换。线性变换的两个参数是变换前的维度，和变换之后的维度
        self.conv_blocks = nn.Sequential(           #nn.sequential{}是一个组成模型的壳子，用来容纳不同的操作
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),        #relu激活函数
            nn.Upsample(scale_factor=2),            #上采样
            nn.Conv2d(128, 64, 3, stride=1, padding=1),#二维卷积
            nn.BatchNorm2d(64, 0.8),               #BN
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),                
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Upsample(size=(224, 224), mode='bilinear',align_corners=False),
            nn.Sigmoid()                               #Tanh激活函数
        )

    def forward(self, z):
        out = self.l1(z)              #l1函数进行的是Linear变换
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) #view是维度变换函数，可以看到out数据变成了四维数据，第一个是batch_size(通过整个的代码，可明白),第二个是channel，第三,四是单张图片的长宽
        img = self.conv_blocks(out)
        return img