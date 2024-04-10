import torch.nn as nn
import torch
from torchsummary import summary


class Autoencodermodel(nn.Module):
    def __init__(self, latent='None'):
        super(Autoencodermodel, self).__init__()
        self.latent = latent
        self.encoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 50, kernel_size=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(50, 150, kernel_size=5),
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.ConvTranspose2d(150, 200, kernel_size=4, stride=2),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.ConvTranspose2d(200, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            
            nn.ConvTranspose2d(256, 200, kernel_size=5, stride=2),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.ConvTranspose2d(200, 180, kernel_size=3, stride=2),
            nn.BatchNorm2d(180),
            nn.ReLU(),
            nn.ConvTranspose2d(180, 150, kernel_size=3, stride=2),
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.ConvTranspose2d(150, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=1),
            nn.Sigmoid()
        )
        if self.latent == 'None':
            self.projection = nn.Identity()


    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        z = self.projection(h.view(-1, 50))

        return h, y, z




if __name__=='__main__':
    model=Autoencodermodel().cuda()
    summary(model,input_size=(768,14,14))

