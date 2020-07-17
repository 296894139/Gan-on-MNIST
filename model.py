import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,out_channel,alpha=0.1):
        super(Discriminator,self).__init__()
        self.out_channel=out_channel
        self.alpha=alpha
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=out_channel,kernel_size=4,stride=3,padding=1,bias=False),
            nn.LeakyReLU(negative_slope=alpha),

            nn.Conv2d(in_channels=out_channel,out_channels=out_channel*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(negative_slope=alpha),

            nn.Conv2d(in_channels=out_channel*2,out_channels=out_channel*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(4*out_channel),
            nn.LeakyReLU(negative_slope=alpha),

            nn.Conv2d(in_channels=out_channel*4,out_channels=out_channel*8,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(8*out_channel),
            nn.LeakyReLU(negative_slope=alpha),

            nn.Conv2d(in_channels=out_channel*8,out_channels=1,kernel_size=4,stride=1,padding=0,bias=False),
            nn.Sigmoid()

        )

    def forward(self,x):
        return self.layers.forward(x).view(-1)



class Generator(nn.Module):

    def __init__(self, in_size, channel):
        super(Generator, self).__init__()
        self.in_size = in_size
        self.channel = channel
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_size, channel * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel * 8),
            nn.ReLU(),


            nn.ConvTranspose2d(8 * channel, 4 * channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel * 4),
            nn.ReLU(),


            nn.ConvTranspose2d(4 * channel, 2 * channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(),


            nn.ConvTranspose2d(2 * channel, channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),


            nn.ConvTranspose2d(channel, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
